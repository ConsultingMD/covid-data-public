import datetime
import pathlib
import re
from typing import MutableMapping

import pandas as pd
import pytz


UNEXPECTED_COLUMNS_MESSAGE = "DataFrame columns do not match expected fields"


def load_county_fips_data(fips_csv: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(fips_csv, dtype={"fips": str})
    df["fips"] = df.fips.str.zfill(5)
    return df


def rename_fields(df, fields, already_transformed_fields, log) -> pd.DataFrame:
    """Return df with columns renamed according to fields, logging and dropping unexpected columns."""
    extra_fields = set(df.columns) - set(fields) - already_transformed_fields
    missing_fields = set(fields) - set(df.columns)
    if extra_fields or missing_fields:
        # If this warning happens in a test you may need to edit the sample data in test/data
        # to make sure all the expected fields appear in the sample.
        log.warning(
            UNEXPECTED_COLUMNS_MESSAGE, extra_fields=extra_fields, missing_fields=missing_fields,
        )
    if extra_fields:
        print("-- Add the following lines to the approriate Fields enum --")
        for extra_field in extra_fields:
            enum_name = re.sub(r"(?<!^)(?=[A-Z])", "_", extra_field).upper()
            print(f'    {enum_name} = "{extra_field}", None')
        print("-- end of suggested new Fields --")
    rename: MutableMapping[str, str] = {f: f for f in already_transformed_fields}
    for col in df.columns:
        field = fields.get(col)
        if field and field.common_field:
            if field.value in rename:
                raise AssertionError(f"Field {repr(field)} misconfigured")
            rename[field.value] = field.common_field.value
    # Copy only columns in `rename.keys()` to a new DataFrame and rename.
    df = df.loc[:, list(rename.keys())].rename(columns=rename)
    return df


def load_census_state(census_state_path: pathlib.Path) -> pd.DataFrame:
    # By default pandas will parse the numeric values in the STATE column as ints but FIPS are two character codes.
    state_df = pd.read_csv(census_state_path, delimiter="|", dtype={"STATE": str})
    state_df.rename(
        columns={"STUSAB": "state", "STATE": "fips", "STATE_NAME": "state_name"}, inplace=True,
    )
    return state_df


def extract_state_fips(fips: str) -> str:
    """Extracts the state FIPS code from a county or state FIPS code."""
    return fips[:2]


def version_timestamp():
    """Returns a Pacific timezone timestamp for use in version.txt files."""
    pacific = pytz.timezone("US/Pacific")
    d = datetime.datetime.now(pacific)
    return d.strftime("%A %b %d %I:%M:%S %p %Z")
