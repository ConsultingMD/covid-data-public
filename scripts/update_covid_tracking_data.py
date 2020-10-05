import dataclasses
import enum
import json
import logging
import datetime
import pathlib
from typing import Optional
from typing import Tuple

import click
import pytz
import requests
import pandas as pd
import numpy as np
import structlog

from covidactnow.datapublic import common_df
from covidactnow.datapublic import common_init
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldNameAndCommonField
from covidactnow.datapublic.common_fields import GetByValueMixin
from scripts import helpers
from scripts import test_positivity

DATA_ROOT = pathlib.Path(__file__).parent.parent / "data"
COVID_TRACKING_ROOT = DATA_ROOT / "covid-tracking"
LOCAL_JSON_PATH = COVID_TRACKING_ROOT / "states.json"
TIMESERIES_CSV_PATH = COVID_TRACKING_ROOT / "timeseries.csv"
TEST_POSITIVITY_FILENAME = "test-positivity-all-methods-wide-dates.csv"
HISTORICAL_STATE_DATA_URL = "http://covidtracking.com/api/states/daily"

ICU_HOSPITALIZED_MISMATCH_WARNING_MESSAGE = (
    "Removed ICU current where it is more than hospitalized current"
)


_logger = logging.getLogger(__name__)


def update_local_json():
    _logger.info("Fetching JSON")
    response = requests.get(HISTORICAL_STATE_DATA_URL)
    LOCAL_JSON_PATH.write_bytes(response.content)


def load_local_json() -> pd.DataFrame:
    _logger.info("Reading local JSON")
    return pd.DataFrame(json.load(LOCAL_JSON_PATH.open("rb")))


class CovidTrackingDataUpdater(object):
    """Updates the covid tracking data."""

    @property
    def output_path(self) -> pathlib.Path:
        return COVID_TRACKING_ROOT / "covid_tracking_states.csv"

    @property
    def version_path(self) -> pathlib.Path:
        return COVID_TRACKING_ROOT / "version.txt"

    @staticmethod
    def _stamp():
        #  String of the current date and time.
        #  So that we're consistent about how we mark these
        pacific = pytz.timezone("US/Pacific")
        d = datetime.datetime.now(pacific)
        return d.strftime("%A %b %d %I:%M:%S %p %Z")

    def update(self):
        _logger.info("Updating Covid Tracking data.")
        df = load_local_json()

        # Removing CT state testing data from three days where numbers were incomplete or negative.
        is_ct = df.state == "CT"
        # TODO(chris): Covid tracking dates are in a weird format, standardize date format.
        dates_to_remove = ["20200717", "20200718", "20200719"]
        df.loc[is_ct & df.date.isin(dates_to_remove), ["negative", "positive"]] = None

        df.to_csv(self.output_path, index=False)

        version_path = self.version_path
        version_path.write_text(f"Updated at {self._stamp()}\n")


@enum.unique
class Fields(GetByValueMixin, FieldNameAndCommonField, enum.Enum):
    # ISO 8601 date of when these values were valid.
    DATE_CHECKED = "dateChecked", None
    STATE = "state", CommonFields.STATE
    # Total cumulative positive test results.
    POSITIVE_TESTS = "positive", CommonFields.POSITIVE_TESTS
    # Increase from the day before.
    POSITIVE_INCREASE = "positiveIncrease", None
    # Total cumulative negative test results.
    NEGATIVE_TESTS = "negative", CommonFields.NEGATIVE_TESTS
    # Increase from the day before.
    NEGATIVE_INCREASE = "negativeIncrease", None
    # Total cumulative number of people hospitalized.
    TOTAL_HOSPITALIZED = "hospitalized", CommonFields.CUMULATIVE_HOSPITALIZED
    # Total cumulative number of people hospitalized.
    CURRENT_HOSPITALIZED = "hospitalizedCurrently", CommonFields.CURRENT_HOSPITALIZED
    # Increase from the day before.
    HOSPITALIZED_INCREASE = "hospitalizedIncrease", None
    # Total cumulative number of people that have died.
    DEATHS = "death", CommonFields.DEATHS
    # Increase from the day before.
    DEATH_INCREASE = "deathIncrease", None
    # Tests that have been submitted to a lab but no results have been reported yet.
    PENDING = "pending", None
    # Calculated value (positive + negative) of total test results.
    TOTAL_TEST_RESULTS = "totalTestResults", None
    # Increase from the day before.
    TOTAL_TEST_RESULTS_INCREASE = "totalTestResultsIncrease", None

    IN_ICU_CURRENTLY = "inIcuCurrently", CommonFields.CURRENT_ICU
    IN_ICU_CUMULATIVE = "inIcuCumulative", CommonFields.CUMULATIVE_ICU

    ON_VENTILATOR_CURRENTLY = "onVentilatorCurrently", CommonFields.CURRENT_VENTILATED
    TOTAL_ON_VENTILATOR = "onVentilatorCumulative", None

    DATE = "date", None
    FIPS = "fips", CommonFields.FIPS

    TOTAL_TESTS_PEOPLE_ANTIBODY = "totalTestsPeopleAntibody", None
    DATE_MODIFIED = "dateModified", None
    NEGATIVE_SCORE = "negativeScore", None
    POS_NEG = "posNeg", None
    DEATH_CONFIRMED = "deathConfirmed", None
    DEATH_PROBABLE = "deathProbable", None
    TOTAL_TESTS_ANTIBODY = "totalTestsAntibody", None
    HOSPITALIZED_CUMULATIVE = "hospitalizedCumulative", None
    TOTAL_TESTS_ANTIGEN = "totalTestsAntigen", None
    POSITIVE_TESTS_PEOPLE_ANTIBODY = "positiveTestsPeopleAntibody", None
    TOTAL_TEST_RESULTS_SOURCE = "totalTestResultsSource", None
    CHECK_TIME_ET = "checkTimeEt", None
    POSITIVE_TESTS_ANTIGEN = "positiveTestsAntigen", None
    TOTAL_TESTS_PEOPLE_VIRAL = "totalTestsPeopleViral", None
    TOTAL_TESTS_VIRAL = "totalTestsViral", None
    RECOVERED = "recovered", None
    NEGATIVE_TESTS_ANTIBODY = "negativeTestsAntibody", None
    COMMERCIAL_SCORE = "commercialScore", None
    POSITIVE_CASES_VIRAL = "positiveCasesViral", None
    SCORE = "score", None
    LAST_UPDATE_ET = "lastUpdateEt", None
    NEGATIVE_TESTS_PEOPLE_ANTIBODY = "negativeTestsPeopleAntibody", None
    TOTAL = "total", None
    HASH = "hash", None
    DATA_QUALITY_GRADE = "dataQualityGrade", None
    NEGATIVE_REGULAR_SCORE = "negativeRegularScore", None
    POSITIVE_TESTS_ANTIBODY = "positiveTestsAntibody", None
    POSITIVE_TESTS_VIRAL = "positiveTestsViral", None
    TOTAL_TEST_ENCOUNTERS_VIRAL = "totalTestEncountersViral", None
    TOTAL_TESTS_PEOPLE_ANTIGEN = "totalTestsPeopleAntigen", None
    NEGATIVE_TESTS_VIRAL = "negativeTestsViral", None
    GRADE = "grade", None
    POSITIVE_SCORE = "positiveScore", None
    POSITIVE_TESTS_PEOPLE_ANTIGEN = "positiveTestsPeopleAntigen", None
    PROBABLE_CASES = "probableCases", None


TEST_POSITIVITY_METHODS = [
    test_positivity.Method(
        "method1", Fields.POSITIVE_CASES_VIRAL, Fields.TOTAL_TEST_ENCOUNTERS_VIRAL
    ),
    test_positivity.Method("method2", Fields.POSITIVE_TESTS_VIRAL, Fields.TOTAL_TESTS_VIRAL),
    test_positivity.Method("method3", Fields.POSITIVE_CASES_VIRAL, Fields.TOTAL_TESTS_VIRAL),
    test_positivity.Method("method4", Fields.POSITIVE_TESTS, Fields.TOTAL_TESTS_VIRAL),
    test_positivity.Method("method5", Fields.POSITIVE_CASES_VIRAL, Fields.TOTAL_TESTS_PEOPLE_VIRAL),
    test_positivity.Method("method6", Fields.POSITIVE_CASES_VIRAL, Fields.TOTAL_TEST_RESULTS),
]


@dataclasses.dataclass
class RegionalDataset:
    """Represents a set of timeseries and provenance information.

    This is similar to MultiRegionTimeseriesDataset in covid-data-model but without the legacy of
    TimeseriesDataset and LatestValuesDataset. Eventually the classes may be merged into one.
    """

    common_timeseries: pd.DataFrame
    common_timeseries_provenance: Optional[pd.Series]

    def __post_init__(self):
        # Some integrity checks
        # TODO(tom): Change to LOCATION_ID when migrated
        assert CommonFields.FIPS in self.common_timeseries.columns
        assert CommonFields.DATE in self.common_timeseries.columns
        assert self.common_timeseries.index.is_unique
        assert self.common_timeseries.index.is_monotonic_increasing
        assert self.common_timeseries.index.names == [None]

        if self.common_timeseries_provenance:
            assert self.common_timeseries_provenance.index.names == [
                CommonFields.LOCATION_ID,
                CommonFields.VARIABLE,
            ]

    def write_csv(self, csv_path: pathlib.Path, log):
        common_df.write_csv(self.common_timeseries, csv_path, log)


def transform(
    df: pd.DataFrame, calculate_test_positivity: bool = False
) -> Tuple[RegionalDataset, Optional[test_positivity.AllMethods]]:
    """Transforms data from load_local_json to the common fields."""
    log = structlog.get_logger()

    # Removing CT state testing data from three days where numbers were incomplete or negative.
    is_ct = df.state == "CT"
    dates_to_remove = ["20200717", "20200718", "20200719"]
    df.loc[is_ct & df.date.isin(dates_to_remove), ["negative", "positive"]] = np.nan

    df[CommonFields.DATE] = pd.to_datetime(df[Fields.DATE], format="%Y%m%d")

    if calculate_test_positivity:
        all_method = test_positivity.AllMethods.run(
            df, TEST_POSITIVITY_METHODS, 7, 14, Fields.STATE
        )
        all_method.all_methods_timeseries.to_csv()
        # TODO(tom): Add test positivty to df and provenance.
    else:
        all_method = None

    # Removing bad data from Delaware.
    # Once that is resolved we can remove this while keeping the assert below.
    icu_mask = df[Fields.IN_ICU_CURRENTLY] > df[Fields.CURRENT_HOSPITALIZED]
    if icu_mask.any():
        df.loc[icu_mask, Fields.IN_ICU_CURRENTLY] = np.nan
        log.warning(
            ICU_HOSPITALIZED_MISMATCH_WARNING_MESSAGE,
            lines_changed=icu_mask.sum(),
            unique_states=df[icu_mask]["state"].nunique(),
        )

    # Current Sanity Check and Filter for In ICU.
    # This should fail for Delaware right now unless we patch it.
    # The 'not any' style is to deal with comparisons to np.nan.
    assert not (
        df[Fields.IN_ICU_CURRENTLY] > df[Fields.CURRENT_HOSPITALIZED]
    ).any(), "IN_ICU_CURRENTLY field is greater than CURRENT_HOSPITALIZED"

    already_transformed_fields = {Fields.DATE}

    df = helpers.rename_fields(df, Fields, already_transformed_fields, log)

    df[CommonFields.COUNTRY] = "USA"

    states_binary_mask = df[CommonFields.FIPS].str.len() == 2
    if not states_binary_mask.all():
        log.warning("Ignoring unexpected non-state regions")
        df = df.loc[states_binary_mask, :]

    df[CommonFields.AGGREGATE_LEVEL] = "state"

    return RegionalDataset(df, None), all_method


@click.command()
@click.option("--replace-local-mirror/--no-replace-local-mirror", default=True)
@click.option("--calculate-test-positivity/--no-calculate-test-positivity", default=False)
def main(replace_local_mirror: bool, calculate_test_positivity: bool):
    logging.basicConfig(level=logging.INFO)
    common_init.configure_logging()

    if replace_local_mirror:
        update_local_json()

    regional_dataset, test_positivity = transform(
        load_local_json(), calculate_test_positivity=calculate_test_positivity
    )
    regional_dataset.write_csv(TIMESERIES_CSV_PATH, structlog.get_logger())
    if test_positivity:
        test_positivity.write(TIMESERIES_CSV_PATH.with_name(TEST_POSITIVITY_FILENAME))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
