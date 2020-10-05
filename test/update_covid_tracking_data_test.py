from io import StringIO
from typing import Tuple

import pandas as pd
import pytest
import structlog

import scripts.test_positivity
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_test_helpers import to_dict

from scripts import update_covid_tracking_data
from scripts.update_covid_tracking_data import Fields
from scripts.test_positivity import Method

from scripts.helpers import UNEXPECTED_COLUMNS_MESSAGE

from scripts.update_covid_tracking_data import ICU_HOSPITALIZED_MISMATCH_WARNING_MESSAGE


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


def test_transform():
    in_df = common_df.read_csv(
        StringIO(
            "date,state,positive,negative,fips,pending,inIcuCurrently,hospitalizedCurrently\n"
            "20200401,TX,10,1000,48,,10,100\n"
            "20200402,TX,11,1100,48,,15,150\n"
        ),
        set_index=False,
    )
    with structlog.testing.capture_logs() as logs:
        out_df = update_covid_tracking_data.transform(in_df, calculate_test_positivity=False)[
            0
        ].common_timeseries

    expected_df = common_df.read_csv(
        StringIO(
            "date,state,country,aggregate_level,positive_tests,negative_tests,fips,current_icu,current_hospitalized\n"
            "2020-04-01,TX,USA,state,10,1000,48,10,100\n"
            "2020-04-02,TX,USA,state,11,1100,48,15,150\n"
        ),
        set_index=False,
    )

    assert to_dict(["fips", "date"], out_df) == to_dict(["fips", "date"], expected_df)

    assert [l["event"] for l in logs] == [
        UNEXPECTED_COLUMNS_MESSAGE,
    ]


def test_transform_icu_greater_than_hospitalized():
    in_df = common_df.read_csv(
        StringIO(
            "date,state,positive,negative,fips,pending,inIcuCurrently,hospitalizedCurrently\n"
            "20200401,TX,10,1000,48,,10,100\n"
            "20200402,TX,11,1100,48,,1500,150\n"
        ),
        set_index=False,
    )
    with structlog.testing.capture_logs() as logs:
        out_df = update_covid_tracking_data.transform(in_df, calculate_test_positivity=False)[
            0
        ].common_timeseries

    expected_df = common_df.read_csv(
        StringIO(
            "date,state,country,aggregate_level,positive_tests,negative_tests,fips,current_icu,current_hospitalized\n"
            "2020-04-01,TX,USA,state,10,1000,48,10,100\n"
            "2020-04-02,TX,USA,state,11,1100,48,,150\n"
        ),
        set_index=False,
    )

    assert to_dict(["fips", "date"], out_df) == to_dict(["fips", "date"], expected_df)

    assert [l["event"] for l in logs] == [
        ICU_HOSPITALIZED_MISMATCH_WARNING_MESSAGE,
        UNEXPECTED_COLUMNS_MESSAGE,
    ]


def test_calculate_all_test_positivity():
    in_df = update_covid_tracking_data.load_local_json()
    with structlog.testing.capture_logs() as logs:
        regional_data, all_test_positivity_methods = update_covid_tracking_data.transform(
            in_df, calculate_test_positivity=True
        )
    assert not regional_data.common_timeseries.empty
    # Make sure there is at least one real value in the timeseries
    assert (
        all_test_positivity_methods.all_methods_timeseries.loc[("TX", "method6"), :].notna().any()
    )


def parse_wide_dates(csv_str: str) -> pd.DataFrame:
    """Parses a string with columns for region, variable/provenance followed by dates."""
    df = pd.read_csv(StringIO(csv_str))
    df = df.set_index(list(df.columns[0:2]))
    df.columns = pd.to_datetime(df.columns)
    return df


def _parse_region_ts_and_series(csv_str: str) -> Tuple[pd.DataFrame, pd.Series]:
    df_full = parse_wide_dates(csv_str)
    second_column = df_full.index.names[1]

    return (
        df_full.reset_index(second_column, drop=True),
        df_full.reset_index(second_column)[second_column],
    )


def test_positivity():
    in_df = common_df.read_csv(
        StringIO(
            "date,state,positive,positiveTestsViral,totalTestResults,\n"
            "2020-04-01,AS,0,,100\n"
            "2020-04-02,AS,2,,200\n"
            "2020-04-03,AS,4,,300\n"
            "2020-04-04,AS,6,,400\n"
            "2020-04-01,TX,1,10,100\n"
            "2020-04-02,TX,2,20,200\n"
            "2020-04-03,TX,3,30,300\n"
            "2020-04-04,TX,4,40,400\n"
        ),
        set_index=False,
    )
    methods = [
        Method("method1", Fields.POSITIVE_TESTS_VIRAL, Fields.TOTAL_TEST_RESULTS),
        Method("method2", Fields.POSITIVE_TESTS, Fields.TOTAL_TEST_RESULTS),
    ]
    all_methods = scripts.test_positivity.AllMethods.run(in_df, methods, 3, 14, CommonFields.STATE)

    expected_df = parse_wide_dates(
        "state,variable,2020-04-01,2020-04-02,2020-04-03,2020-04-04\n"
        "AS,method2,,,,0.02\n"
        "TX,method1,,,,0.1\n"
        "TX,method2,,,,0.01\n"
    )
    pd.testing.assert_frame_equal(all_methods.all_methods_timeseries, expected_df, check_like=True)


def test_positivity_recent_days():
    in_df = common_df.read_csv(
        StringIO(
            "date,state,positive,positiveTestsViral,totalTestResults,\n"
            "2020-04-01,AS,0,0,100\n"
            "2020-04-02,AS,2,20,200\n"
            "2020-04-03,AS,4,,300\n"
            "2020-04-04,AS,6,,400\n"
            "2020-04-01,TX,1,10,100\n"
            "2020-04-02,TX,2,20,200\n"
            "2020-04-03,TX,3,30,300\n"
            "2020-04-04,TX,4,40,400\n"
        ),
        set_index=False,
    )
    methods = [
        Method("method1", Fields.POSITIVE_TESTS_VIRAL, Fields.TOTAL_TEST_RESULTS),
        Method("method2", Fields.POSITIVE_TESTS, Fields.TOTAL_TEST_RESULTS),
    ]
    all_methods = scripts.test_positivity.AllMethods.run(in_df, methods, 1, 2, CommonFields.STATE)

    expected_all = parse_wide_dates(
        "state,variable,2020-04-01,2020-04-02,2020-04-03,2020-04-04\n"
        "AS,method1,,0.2,,\n"
        "AS,method2,,0.02,0.02,0.02\n"
        "TX,method1,,0.1,0.1,0.1\n"
        "TX,method2,,0.01,0.01,0.01\n"
    )
    pd.testing.assert_frame_equal(all_methods.all_methods_timeseries, expected_all, check_like=True)
    expected_df, expected_provenance = _parse_region_ts_and_series(
        "state,provenance,2020-04-01,2020-04-02,2020-04-03,2020-04-04\n"
        "AS,method2,,0.02,0.02,0.02\n"
        "TX,method1,,0.1,0.1,0.1\n"
    )
    pd.testing.assert_frame_equal(
        all_methods.test_positivity, expected_df, check_like=True,
    )
    pd.testing.assert_series_equal(all_methods.provenance, expected_provenance)
