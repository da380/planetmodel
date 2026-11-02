"""The registry check: every shipped entry fetched from its host, its hashes
checked, read, and its product held to its protocol.

These tests touch the network and are selected only with `-m network`
(by hand, and by the registry workflow on demand and monthly); a changed
file upstream fails here first.  The cache is `PLANETMODEL_DATA`, or the
platform's.  An entry every one of whose URLs is on a host refusing
GitHub's runners is skipped on GitHub Actions, saying so, and checked
when the tests are run by hand.
"""
import os
from urllib.parse import urlsplit

import pytest

from planetmodel import datasets
from planetmodel.datasets import REGISTRY, Dataset, GridReader
from planetmodel.testing import check_dataset

pytestmark = pytest.mark.network

#: Hosts that refuse GitHub's runners while serving the same request from
#: elsewhere, and how.
REFUSING_RUNNERS = {"ddfe.curtin.edu.au": "answers 403 Forbidden to GitHub's runners"}


def refused_on_runners(entry: Dataset) -> str | None:
    """Why an entry cannot be fetched on GitHub Actions: every URL it is
    served from, mirrors included, is on a host of `REFUSING_RUNNERS`;
    None off GitHub Actions, or where one URL is on another host."""
    if not os.environ.get("GITHUB_ACTIONS"):
        return None
    hosts = {urlsplit(url).hostname for source in entry.sources
             for url in (source.url, *source.mirrors)}
    if not hosts or not hosts <= set(REFUSING_RUNNERS):
        return None
    return "; ".join(f"{host} {REFUSING_RUNNERS[host]}" for host in sorted(hosts))


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_entry_fetches_reads_and_holds_to_its_contract(name):
    entry = REGISTRY[name]
    why = refused_on_runners(entry)
    if why is not None:
        pytest.skip(f"{why}; run the network tests by hand to check {name}")
    if isinstance(entry.reader, GridReader):
        pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    paths = datasets.fetch(entry)
    check_dataset(entry, paths=paths)
