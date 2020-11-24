"""
Data schema shared between code in covid-data-public and covid-data-model repos.
"""
import enum
from typing import Optional


class GetByValueMixin:
    """Mixin making it easy to get an Enum object or None if not found.

    Unlike `YourEnumClass(value)`, the `get` method does not raise `ValueError` when `value`
    is not in the enum.
    """

    @classmethod
    def get(cls, value):
        return cls._value2member_map_.get(value, None)


class ValueAsStrMixin:
    def __str__(self):
        return self.value


class FieldName(str):
    """Common base-class for enums of fields, CSV column names etc"""

    pass


@enum.unique
class CommonFields(GetByValueMixin, ValueAsStrMixin, FieldName, enum.Enum):
    """Common field names shared across different sources of data"""

    FIPS = "fips"

    DATE = "date"

    # In the style of CovidAtlas/Project Li `locationID`. See
    # https://github.com/covidatlas/li/blob/master/docs/reports-v1.md#general-notes
    LOCATION_ID = "location_id"

    # 2 letter state abbreviation, i.e. MA
    STATE = "state"

    COUNTRY = "country"

    COUNTY = "county"

    AGGREGATE_LEVEL = "aggregate_level"

    # Full state name, i.e. Massachusetts
    STATE_FULL_NAME = "state_full_name"

    CASES = "cases"
    DEATHS = "deaths"
    RECOVERED = "recovered"

    # Incidence Values
    NEW_CASES = "new_cases"
    NEW_DEATHS = "new_deaths"
    WEEKLY_NEW_CASES = "weekly_new_cases"
    WEEKLY_NEW_DEATHS = "weekly_new_deaths"

    # Forecast Specific Columns
    MODEL_ABBR = "model_abbr"  # The label of the model used for prediction
    FORECAST_DATE = "forecast_date"  # The prediction made with data up to that date
    QUANTILE = "quantile"  # Prediction Levels

    # Cumulative values
    CUMULATIVE_HOSPITALIZED = "cumulative_hospitalized"
    CUMULATIVE_ICU = "cumulative_icu"

    POSITIVE_TESTS = "positive_tests"
    NEGATIVE_TESTS = "negative_tests"
    TOTAL_TESTS = "total_tests"

    POSITIVE_TESTS_VIRAL = "positive_tests_viral"
    POSITIVE_CASES_VIRAL = "positive_cases_viral"
    TOTAL_TESTS_VIRAL = "total_tests_viral"
    TOTAL_TESTS_PEOPLE_VIRAL = "total_tests_people_viral"
    TOTAL_TEST_ENCOUNTERS_VIRAL = "total_test_encounters_viral"

    # Current values
    CURRENT_ICU = "current_icu"
    CURRENT_HOSPITALIZED = "current_hospitalized"
    CURRENT_VENTILATED = "current_ventilated"

    POPULATION = "population"

    STAFFED_BEDS = "staffed_beds"
    LICENSED_BEDS = "licensed_beds"
    ICU_BEDS = "icu_beds"
    ALL_BED_TYPICAL_OCCUPANCY_RATE = "all_beds_occupancy_rate"
    ICU_TYPICAL_OCCUPANCY_RATE = "icu_occupancy_rate"
    MAX_BED_COUNT = "max_bed_count"
    VENTILATOR_CAPACITY = "ventilator_capacity"

    HOSPITAL_BEDS_IN_USE_ANY = "hospital_beds_in_use_any"
    CURRENT_HOSPITALIZED_TOTAL = "current_hospitalized_total"
    CURRENT_ICU_TOTAL = "current_icu_total"

    CONTACT_TRACERS_COUNT = "contact_tracers_count"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"

    # Ratio of positive tests to total tests, from 0.0 to 1.0
    TEST_POSITIVITY = "test_positivity"


@enum.unique
class PdFields(GetByValueMixin, ValueAsStrMixin, FieldName, enum.Enum):
    """Field names that are used in Pandas but not directly related to COVID metrics"""

    # Identifies the metric or variable name in Panda DataFrames with only one value ('long' layout) or
    # timeseries ('date wide' layout) per row.
    VARIABLE = "variable"
    # Column containing the value in 'long' format DataFrames.
    VALUE = "value"

    PROVENANCE = "provenance"

    # The name of the dataset. This was added to enable having multiple dataset in a
    # single DataFrame while merging test positivity data sources.
    DATASET = "dataset"


# CommonFields used as keys/index columns in timeseries DataFrames.
# I'd like this to be immutable (for example a tuple) but pandas sometimes treats tuples and lists
# differently and many covid-data-model tests fail when it is a tuple.
COMMON_FIELDS_TIMESERIES_KEYS = [CommonFields.FIPS, CommonFields.DATE]


# Fields that are currently expected when representing a region in a DataFrame and CSV. Newer code is expected
# to only depend on the character field FIPS.
COMMON_LEGACY_REGION_FIELDS = [
    CommonFields.FIPS,
    CommonFields.STATE,
    CommonFields.COUNTRY,
    CommonFields.COUNTY,
    CommonFields.AGGREGATE_LEVEL,
]


COMMON_FIELDS_ORDER_MAP = {common: i for i, common in enumerate(CommonFields)}


class FieldNameAndCommonField(FieldName):
    """Represents the original field/column name and CommonField it maps to or None if dropped."""

    def __new__(cls, field_name: str, common_field: Optional[CommonFields]):
        o = super().__new__(cls, field_name)
        o.common_field = common_field
        return o
