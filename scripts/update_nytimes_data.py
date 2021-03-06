from typing import Dict, Tuple, List
import enum
import logging
import datetime
import shutil
import tempfile


import pathlib
import requests
import pandas as pd
import pydantic
import structlog
import click

from covidactnow.datapublic import common_init
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import (
    GetByValueMixin,
    CommonFields,
    COMMON_FIELDS_TIMESERIES_KEYS,
    FieldNameAndCommonField,
)
from scripts import helpers

DATA_ROOT = pathlib.Path(__file__).parent.parent / "data"
_logger = structlog.get_logger(__name__)


STATE_BACKFILLED_CASES = [
    # On 2020-07-24, CT reported a backfill of 440 additional positive cases.
    # https://portal.ct.gov/Office-of-the-Governor/News/Press-Releases/2020/07-2020/Governor-Lamont-Coronavirus-Update-July-24
    ("09", "2020-07-24", 440),
    # https://portal.ct.gov/Office-of-the-Governor/News/Press-Releases/2020/07-2020/Governor-Lamont-Coronavirus-Update-July-29
    ("09", "2020-07-29", 384),
    # http://www.floridahealth.gov/newsroom/2020/09/090120-1112-covid19.pr.html
    ("12", "2020-09-01", 3870),
    # https://twitter.com/DHSWI/status/1301980028209713153?s=20
    # This number is a bit fuzzy, but on 9/3 there were 767 cases.  There were 1400 cases
    # on 9/4 including backfill.  Estimating 800 cases on 9/4.
    ("55", "2020-09-04", 1400 - 800),
    # https://directorsblog.health.azdhs.gov/covid-19-antigen-tests/
    ("04", "2020-09-17", 577),
    # https://twitter.com/AZDHS/status/1306974258124468226
    ("04", "2020-09-18", 764),
    # https://trello.com/c/9EbBRV1Z/472-2020-09-26-north-carolina-change-in-methodology-to-count-cases
    ("37", "2020-09-25", 4544),
    # https://www.al.com/news/2020/10/alabama-adds-3852-covid-cases-after-influx-of-backlogged-data-dating-back-to-june.html
    ("01", "2020-10-23", 2565),
]


COUNTY_BACKFILLED_CASES = [
    # https://www.collincountytx.gov/public_information/news/Pages/08142020_COVIDreport.aspx
    # Collin county had a large backfill of 1175 cases.  It's not known how many of these cases
    # are backfill vs new cases. Assuming that cases are around the previous and next day
    # (132 and 395) subtracting 900 cases to give 275 new cases for that day.
    ("48085", "2020-08-14", 900),
    # https://www.dallasnews.com/news/public-health/2020/08/16/backlog-in-state-reporting-adds-more-than-5000-coronavirus-cases-in-dallas-county/
    # Of the 5,361 cases reported Sunday, 5,195 came from the backlog, according to a news release from Dallas County Judge Clay Jenkins.
    ("48113", "2020-08-16", 5195),
    # https://www.dallascounty.org/Assets/uploads/docs/covid-19/press-releases/august/081720-PressRelease-DallasCountyReports1850AdditionalPositiveCOVID-19Cases.pdf
    # Dallas County has requested the breakdown of backfilled/non-backfilled cases
    # but has not yet received it: https://twitter.com/JudgeClayJ/status/1295836564887678976?s=20
    # 1500 is an estimate that puts daily increases inline with other days (leaving around 300 new cases).
    ("48113", "2020-08-17", 1500),
    # https://www.dallasnews.com/news/public-health/2020/08/18/dallas-county-expects-to-report-about-550-backlogged-coronavirus-cases-237-new-cases/
    ("48113", "2020-08-18", 550),
    # https://www.dallascounty.org/Assets/uploads/docs/covid-19/press-releases/august/082020-PressRelease-DallasCountyReports308AdditionalPositiveCOVID-19Cases.pdf
    ("48113", "2020-08-19", 206),
    # https://www.dallascounty.org/Assets/uploads/docs/covid-19/press-releases/august/082120-PressRelease-DallasCountyReports714AdditionalPositiveCOVID-19Cases.pdf
    ("48113", "2020-08-21", 459),
    # https://www.dallascounty.org/Assets/uploads/docs/covid-19/press-releases/august/082220-PressRelease-DallasCountyReports1086AdditionalPositiveCOVID-19Cases.pdf
    ("48113", "2020-08-22", 862),
    # https://www.dallascounty.org/Assets/uploads/docs/covid-19/press-releases/august/082320-PressRelease-DallasCountyReports1086AdditionalPositiveCOVID-19Cases.pdf
    ("48113", "2020-08-23", 93),
    # https://www.dallascounty.org/Assets/uploads/docs/covid-19/press-releases/august/082420-PressRelease-DallasCountyReports182AdditionalPositiveCOVID-19Cases.pdf
    ("48113", "2020-08-24", 84),
    # https://covid-19-in-jefferson-county-ky-lojic.hub.arcgis.com/
    ("21111", "2020-09-03", 500),
    # https://www.ibj.com/articles/official-covid-19-cases-surpass-20000-in-marion-county/
    # NYT inaccurately showed 411 instead of 426
    ("18097", "2020-09-13", 396),
    # https://www.ksat.com/news/local/2020/09/14/142-new-covid-19-cases-24-deaths-reported-for-san-antonio-bexar-county/
    ("48029", "2020-09-13", 1575),
    # https://harriscounty.maps.arcgis.com/apps/opsdashboard/index.html#/c0de71f8ea484b85bb5efcb7c07c6914
    ("48201", "2020-09-15", 2378),
    # https://trello.com/c/CEkDc3ZO/462-large-tx-backfill-on-9-21
    ("48029", "2020-09-20", 2078),  # Bexar County, TX
    ("48113", "2020-09-20", 306),  # Dallas County, TX
    ("48001", "2020-09-21", 1051),  # Anderson County, TX
    ("48085", "2020-09-21", 3),  # Collin County, TX
    ("48163", "2020-09-21", 298),  # Frio County, TX
    ("48201", "2020-09-21", 328),  # Harris County, TX
    ("48225", "2020-09-21", 1),  # Houston County, TX
    ("48439", "2020-09-21", 125),  # Tarrant County, TX
    ("48013", "2020-09-21", 522),  # Atascosa County, TX
    ("48019", "2020-09-21", 41),  # Bandera County, TX
    ("48057", "2020-09-21", 186),  # Calhoun County, TX
    ("48113", "2020-09-21", 2),  # Dallas County, TX
    ("48127", "2020-09-21", 53),  # Dimmit County, TX
    ("48137", "2020-09-21", 33),  # Edwards County, TX
    ("48171", "2020-09-21", 96),  # Gillespie County, TX
    ("48177", "2020-09-21", 234),  # Gonzales County, TX
    ("48187", "2020-09-21", 1587),  # Guadalupe County, TX
    ("48201", "2020-09-21", 13622),  # Harris County, TX
    ("48239", "2020-09-21", 77),  # Jackson County, TX
    ("48255", "2020-09-21", 181),  # Karnes County, TX
    ("48265", "2020-09-21", 142),  # Kerr County, TX
    ("48271", "2020-09-21", 19),  # Kinney County, TX
    ("48285", "2020-09-21", 252),  # Lavaca County, TX
    ("48355", "2020-09-21", 231),  # Nueces County, TX
    ("48385", "2020-09-21", 12),  # Real County, TX
    ("48493", "2020-09-21", 307),  # Wilson County, TX
    ("48507", "2020-09-21", 52),  # Zavala County, TX
    ("48407", "2020-09-22", 1),  # San Jacinto, TX
    # https://trello.com/c/GKUHUbyK/483-2020-10-03-collin-county-tx-spike-on-daily-new-cases
    ("48201", "2020-10-02", 2438),  # Harris County, TX
    ("48085", "2020-10-03", 1202),  # Collin County, TX
]


def _calculate_county_adjustments(
    data: pd.DataFrame, date: str, backfilled_cases: int, state_fips: str
) -> Dict[str, int]:
    """Calculating number of cases to remove per county, weighted on number of new cases per county.

    Weighting on number of new cases per county gives a reasonable measure of where the backfilled
    cases ended up.

    Args:
        data: Input Data.
        date: Date of backfill.
        backfilled_cases: Number of backfilled cases.
        state_fips: FIPS code for state.

    Returns: Dictionary of estimated fips -> backfilled cases.
    """
    is_state = data[CommonFields.FIPS].str.match(f"{state_fips}[0-9][0-9][0-9]")
    is_not_unknown = data[CommonFields.FIPS] != f"{state_fips}999"
    if not (is_not_unknown & is_state).any():
        return {}

    fields = [CommonFields.DATE, CommonFields.FIPS, CommonFields.CASES]
    cases = (
        data.loc[is_state & is_not_unknown, fields]
        .set_index([CommonFields.FIPS, CommonFields.DATE])
        .sort_index()
    )
    cases = cases.diff().reset_index(level=1)
    cases_on_date = cases[cases.date == date]["cases"]
    # For states with more counties, rounding could lead to the sum of the counties diverging from
    # the backfilled cases count.
    return (cases_on_date / cases_on_date.sum() * backfilled_cases).round().to_dict()


def remove_state_backfilled_cases(
    data: pd.DataFrame, backfilled_cases: List[Tuple[str, str, int]]
) -> pd.DataFrame:
    """Removes reported backfilled cases from case totals.

    Args:
        data: Data
        backfilled_cases: List of backfilled case info.

    Returns: Updated data frame.
    """
    for state_fips, date, cases in backfilled_cases:
        adjustments = _calculate_county_adjustments(data, date, cases, state_fips)
        is_on_or_after_date = data[CommonFields.DATE] >= date
        for fips, count in adjustments.items():
            is_fips_data_after_date = is_on_or_after_date & (data[CommonFields.FIPS] == fips)
            data.loc[is_fips_data_after_date, CommonFields.CASES] -= int(count)

        # Remove state counts also.
        is_fips_data_after_date = is_on_or_after_date & (data[CommonFields.FIPS] == state_fips)
        if is_fips_data_after_date.any():
            data.loc[is_fips_data_after_date, CommonFields.CASES] -= cases

    return data


def remove_county_backfilled_cases(
    data: pd.DataFrame, backfilled_cases: List[Tuple[str, str, int]]
) -> pd.DataFrame:
    """Removes reported county backfilled cases from case totals.

    Args:
        data: Data
        backfilled_cases: List of backfilled case info.

    Returns: Updated data frame.
    """
    for county_fips, date, cases in backfilled_cases:
        is_on_or_after_date = data[CommonFields.DATE] >= date
        is_fips_data_after_date = is_on_or_after_date & (data[CommonFields.FIPS] == county_fips)
        data.loc[is_fips_data_after_date, CommonFields.CASES] -= cases

        # Remove county count from state counts as well
        state_fips = helpers.extract_state_fips(county_fips)
        is_fips_data_after_date = is_on_or_after_date & (data[CommonFields.FIPS] == state_fips)
        if is_fips_data_after_date.any():
            data.loc[is_fips_data_after_date, CommonFields.CASES] -= cases

    return data


def _remove_ma_county_zeroes_data(
    data: pd.DataFrame,
    county_reporting_stopped_date="2020-08-11",
    county_reporting_restart_date="2020-08-18",
):
    """Removes county data for mass where cases are not increasing due to data reporting change.

    Massachussetts stopped reporting county case data after 8/11.  This code removes data for those
    days, treating the data as missing rather than a count of 0.

    Args:
        data: Data to clean up.
        county_reporting_stopped_date: Date to start checking for no case count increases.
        county_reporting_restart_date: Date that MA county reporting started up again.

    Returns: Data with Mass county data properly cleaned up.
    """
    # Sorting on fips and date to ensure the diff is applied in ascending date order below.
    data = data.sort_values([CommonFields.FIPS, CommonFields.DATE])

    is_county = data[CommonFields.AGGREGATE_LEVEL] == "county"
    is_ma = data[CommonFields.STATE] == "MA"
    is_during_reporting_lull = data[CommonFields.DATE].between(
        county_reporting_stopped_date, county_reporting_restart_date
    )
    is_ma_county_after_reporting = is_county & is_ma & is_during_reporting_lull
    ma_county_data = data.loc[is_ma_county_after_reporting]
    cases_to_remove = ma_county_data.groupby(Fields.FIPS)[CommonFields.CASES].diff() == 0
    _logger.info("Removing stale MA county cases", num_records=sum(cases_to_remove))
    return pd.concat(
        [data.loc[~is_ma_county_after_reporting], ma_county_data.loc[~cases_to_remove]]
    )


@enum.unique
class Fields(GetByValueMixin, FieldNameAndCommonField, enum.Enum):
    DATE = "date", CommonFields.DATE
    COUNTY = "county", CommonFields.COUNTY
    STATE_FULL_NAME = "state", CommonFields.STATE_FULL_NAME
    FIPS = "fips", CommonFields.FIPS
    CASES = "cases", CommonFields.CASES
    DEATHS = "deaths", CommonFields.DEATHS


class NYTimesUpdater(pydantic.BaseModel):
    """Updates NYTimes data set"""

    COUNTY_CSV_FILENAME = "us-counties.csv"
    STATE_CSV_FILENAME = "us-states.csv"
    VERSION_FILENAME = "version.txt"

    NYTIMES_MASTER_API_URL_STATES = (
        "https://api.github.com/repos/nytimes/covid-19-data/contents/us-states.csv"
    )
    NYTIMES_RAW_BASE_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master"

    raw_data_root: pathlib.Path

    timeseries_output_path: pathlib.Path

    state_census_path: pathlib.Path

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def make_with_data_root(cls, data_root: pathlib.Path) -> "NYTimesUpdater":
        return cls(
            raw_data_root=data_root / "cases-nytimes",
            timeseries_output_path=data_root / "cases-nytimes" / "timeseries-common.csv",
            state_census_path=data_root / "misc" / "state.txt",
        )

    # This is the only one being accessed currently, also available are us-states.csv and us.csv
    _DATA_TARGET_FILES = [COUNTY_CSV_FILENAME, STATE_CSV_FILENAME]

    def _get_repo_archive_url(self, git_sha: str) -> str:
        return f"https://github.com/nytimes/covid-19-data/archive/{git_sha}.zip"

    @property
    def county_url(self):
        return f"{self.NYTIMES_RAW_BASE_URL}/{self.COUNTY_CSV_FILENAME}"

    @property
    def state_url(self):
        return f"{self.NYTIMES_RAW_BASE_URL}/{self.STATE_CSV_FILENAME}"

    @property
    def county_path(self) -> pathlib.Path:
        return self.raw_data_root / self.COUNTY_CSV_FILENAME

    @property
    def state_path(self) -> pathlib.Path:
        return self.raw_data_root / self.STATE_CSV_FILENAME

    def get_master_commit_sha(self) -> str:
        # Getting the master commit of the states file as the master URL may contain
        # changes not applicable to the files we download.
        r = requests.get(self.NYTIMES_MASTER_API_URL_STATES)
        return r.json()["sha"]

    def write_version_file(self, git_sha) -> None:
        stamp = datetime.datetime.utcnow().isoformat()
        version_path = self.raw_data_root / "version.txt"
        with version_path.open("w+") as vf:
            vf.write(f"{git_sha}\n")
            vf.write(f"Updated on {stamp}")

    def read_version_file_sha(self) -> str:
        version_path = self.raw_data_root / "version.txt"
        with version_path.open("r") as vf:
            return vf.readline().rstrip("\n")

    def is_new_data_available(self):
        """Check to see if the sha for the data files have updated."""
        current_sha = self.read_version_file_sha()

        new_sha = self.get_master_commit_sha()
        return current_sha != new_sha

    def update_source_data(self):
        git_sha = self.get_master_commit_sha()
        _logger.info(f"Updating version file with nytimes revision {git_sha}")
        state_data = requests.get(self.state_url).content
        self.state_path.write_bytes(state_data)

        county_data = requests.get(self.county_url).content
        self.county_path.write_bytes(county_data)
        self.write_version_file(git_sha)

    def load_state_and_county_data(self) -> pd.DataFrame:
        """Loads state and county data in one dataset, renaming fields to common field names. """
        _logger.info("Updating NYTimes dataset.")
        # Able to use common_df here because the NYTimes raw files include fips and date.
        county_data = common_df.read_csv(self.county_path).reset_index()
        county_data = helpers.rename_fields(county_data, Fields, set(), _logger)
        county_data[CommonFields.AGGREGATE_LEVEL] = "county"

        # Able to use common_df here because the NYTimes raw files include fips and date.
        state_data = common_df.read_csv(self.state_path).reset_index()
        state_data = helpers.rename_fields(state_data, Fields, set(), _logger)
        state_data[CommonFields.AGGREGATE_LEVEL] = "state"

        return pd.concat([county_data, state_data])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        census_data = helpers.load_census_state(self.state_census_path).set_index("state_name")

        # Renaming Virgin Islands to match full name from census_data
        data[CommonFields.STATE_FULL_NAME] = data[CommonFields.STATE_FULL_NAME].replace(
            "Virgin Islands", "U.S. Virgin Islands"
        )
        data[CommonFields.STATE] = data[CommonFields.STATE_FULL_NAME].map(census_data["state"])
        data[CommonFields.COUNTRY] = "USA"

        # Rename new york city to new york county and assign it to New York County FIPS
        ny_mask = data[CommonFields.COUNTY].astype(str) == "New York City"
        data.loc[ny_mask, CommonFields.COUNTY] = "New York County"
        data.loc[ny_mask, CommonFields.FIPS] = "36061"

        data = remove_state_backfilled_cases(data, STATE_BACKFILLED_CASES)
        data = remove_county_backfilled_cases(data, COUNTY_BACKFILLED_CASES)

        data = _remove_ma_county_zeroes_data(data)

        no_fips = data[CommonFields.FIPS].isna()
        if no_fips.any():
            _logger.error(
                "Rows without fips", no_fips=data.loc[no_fips, CommonFields.COUNTY].value_counts()
            )
            data = data.loc[~no_fips, :]

        return data


@click.command()
@click.option("--check-for-new-data", is_flag=True)
@click.option("--fetch/--no-fetch", default=True)
def main(check_for_new_data: bool, fetch: bool):
    common_init.configure_logging()
    transformer = NYTimesUpdater.make_with_data_root(DATA_ROOT)

    if check_for_new_data:
        if not transformer.is_new_data_available():
            raise Exception("No new data available")
        _logger.info("New data available")
        return

    if fetch:
        _logger.info("Fetching new data.")
        transformer.update_source_data()

    data = transformer.load_state_and_county_data()
    data = transformer.transform(data)
    common_df.write_csv(data, transformer.timeseries_output_path, _logger)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
