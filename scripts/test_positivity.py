import dataclasses
from itertools import chain
from typing import List

import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields, FieldName


@dataclasses.dataclass
class Method:
    """A method of calculating test positivity"""

    name: str
    numerator: FieldName
    denominator: FieldName

    def calculate(self, delta_df: pd.DataFrame) -> pd.DataFrame:
        assert delta_df.columns.names == [CommonFields.DATE]
        assert delta_df.index.names == ["variable", CommonFields.STATE]
        # delta_df has the field name as the first level of the index. delta_df.loc[field, :] returns a
        # DataFrame without the field label so operators such as `/` are calculated for each
        # region/state and date.
        return delta_df.loc[self.numerator, :] / delta_df.loc[self.denominator, :]


@dataclasses.dataclass
class AllMethods:
    """The result of calculating all methods for all regions"""

    all_methods_timeseries: pd.DataFrame
    test_positivity: pd.DataFrame
    provenance: pd.Series

    @staticmethod
    def run(
        df: pd.DataFrame,
        methods: List[Method],
        diff_days: int,
        recent_days: int,
        region_field: FieldName,
        date_field: FieldName = CommonFields.DATE,
    ) -> "AllMethods":
        key_cols = [region_field, date_field]
        ts_value_cols = list(
            set(chain.from_iterable((method.numerator, method.denominator) for method in methods))
        )

        input_long = (
            df.loc[:, key_cols + ts_value_cols]
            .melt(id_vars=key_cols, value_vars=ts_value_cols)
            .dropna()
            .set_index(["variable", region_field, date_field])["value"]
            .apply(pd.to_numeric)
        )
        start_date = df[CommonFields.DATE].min()
        end_date = df[CommonFields.DATE].max()
        input_date_range = pd.date_range(start=start_date, end=end_date)
        recent_date_range = pd.date_range(end=end_date, periods=recent_days).intersection(
            input_date_range
        )
        input_wide = (
            input_long.unstack(date_field)
            .reindex(columns=input_date_range)
            .rename_axis(columns=date_field)
        )
        # This calculates the difference only when the cumulative value is a real value `diff_days` apart.
        # It looks like our input data has few or no holes so this works well enough.
        diff_df = input_wide.diff(periods=diff_days, axis=1)

        all_wide = (
            pd.concat(
                {method.name: method.calculate(diff_df) for method in methods}, names=["variable"]
            )
            .reorder_levels([region_field, "variable"])
            # Drop empty timeseries
            .dropna("index", "all")
            .sort_index()
        )

        method_cat_type = pd.CategoricalDtype(
            categories=[method.name for method in methods], ordered=True
        )

        has_recent_data = all_wide.loc[:, recent_date_range].notna().any(axis=1)
        all_recent_data = all_wide.loc[has_recent_data, :].reset_index()
        all_recent_data["variable"] = all_recent_data["variable"].astype(method_cat_type)
        first = all_recent_data.groupby(region_field).first()
        provenance = first["variable"].astype(str).rename("provenance")
        positivity = first.drop(columns=["variable"])

        return AllMethods(
            all_methods_timeseries=all_wide, test_positivity=positivity, provenance=provenance
        )
