import enum
import json
import logging
import datetime
import pathlib

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

DATA_ROOT = pathlib.Path(__file__).parent.parent / "data"
COVID_TRACKING_ROOT = DATA_ROOT / "covid-tracking"
LOCAL_JSON_PATH = COVID_TRACKING_ROOT / "states.json"
TIMESERIES_CSV_PATH = COVID_TRACKING_ROOT / "timeseries.csv"
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
    TOTAL_TEST_RESULTS = "totalTestResults", CommonFields.TOTAL_TESTS
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
    TOTAL_TESTS_PEOPLE_VIRAL = "totalTestsPeopleViral", CommonFields.TOTAL_TESTS_PEOPLE_VIRAL
    TOTAL_TESTS_VIRAL = "totalTestsViral", CommonFields.TOTAL_TESTS_VIRAL
    RECOVERED = "recovered", None
    NEGATIVE_TESTS_ANTIBODY = "negativeTestsAntibody", None
    COMMERCIAL_SCORE = "commercialScore", None
    POSITIVE_CASES_VIRAL = "positiveCasesViral", CommonFields.POSITIVE_CASES_VIRAL
    SCORE = "score", None
    LAST_UPDATE_ET = "lastUpdateEt", None
    NEGATIVE_TESTS_PEOPLE_ANTIBODY = "negativeTestsPeopleAntibody", None
    TOTAL = "total", None
    HASH = "hash", None
    DATA_QUALITY_GRADE = "dataQualityGrade", None
    NEGATIVE_REGULAR_SCORE = "negativeRegularScore", None
    POSITIVE_TESTS_ANTIBODY = "positiveTestsAntibody", None
    POSITIVE_TESTS_VIRAL = "positiveTestsViral", CommonFields.POSITIVE_TESTS_VIRAL
    TOTAL_TEST_ENCOUNTERS_VIRAL = (
        "totalTestEncountersViral",
        CommonFields.TOTAL_TEST_ENCOUNTERS_VIRAL,
    )
    TOTAL_TESTS_PEOPLE_ANTIGEN = "totalTestsPeopleAntigen", None
    NEGATIVE_TESTS_VIRAL = "negativeTestsViral", None
    GRADE = "grade", None
    POSITIVE_SCORE = "positiveScore", None
    POSITIVE_TESTS_PEOPLE_ANTIGEN = "positiveTestsPeopleAntigen", None
    PROBABLE_CASES = "probableCases", None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms data from load_local_json to the common fields."""
    log = structlog.get_logger()

    # Removing CT state testing data from three days where numbers were incomplete or negative.
    is_ct = df.state == "CT"
    dates_to_remove = ["20200717", "20200718", "20200719"]
    df.loc[is_ct & df.date.isin(dates_to_remove), ["negative", "positive"]] = np.nan

    df[CommonFields.DATE] = pd.to_datetime(df[Fields.DATE], format="%Y%m%d")

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

    return df


@click.command()
@click.option("--replace-local-mirror/--no-replace-local-mirror", default=True)
@click.option("--generate-common-csv/--no-generate-common-csv", default=True)
def main(replace_local_mirror: bool, generate_common_csv: bool):
    logging.basicConfig(level=logging.INFO)
    common_init.configure_logging()

    if replace_local_mirror:
        update_local_json()

    CovidTrackingDataUpdater().update()

    if generate_common_csv:
        common_df.write_csv(
            transform(load_local_json()), TIMESERIES_CSV_PATH, structlog.get_logger(),
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
