
import pandas as pd
from kdiagram.core.property import PandasDataHandlers

def test_parsers_and_writers_presence():
    h = PandasDataHandlers()
    assert ".csv" in h.parsers and callable(h.parsers[".csv"])
    df = pd.DataFrame({"a": [1, 2]})
    w = h.writers(df)
    assert ".csv" in w and callable(w[".csv"])
