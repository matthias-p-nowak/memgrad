import numpy as np
import pandas as pd
import plotly.express as px

df = pd.DataFrame({"x": np.linspace(0, 10, 200), "y": np.sin(np.linspace(0, 10, 200))})
fig = px.line(df, x="x", y="y")
fig.show()
