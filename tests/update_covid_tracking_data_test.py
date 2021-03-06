from io import StringIO

import pytest
import structlog

from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_test_helpers import to_dict

from scripts import update_covid_tracking_data

from scripts import helpers
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
        out_df = update_covid_tracking_data.transform(in_df)

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
        helpers.MISSING_COLUMNS_MESSAGE,
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
        out_df = update_covid_tracking_data.transform(in_df)

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
        helpers.MISSING_COLUMNS_MESSAGE,
    ]
