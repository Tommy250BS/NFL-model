from stacking_model import run_backtest
model, alpha, results = run_backtest()

import pandas as pd
from stacking_model import FEATURE_COLS
imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(imp)