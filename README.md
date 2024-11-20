# Data-Science-Covid-forecast
COVID-19 Forecast for Poland (December)  This project aims to forecast daily new COVID-19 cases in Poland for December using time-series analysis. The data, sourced from "Our World in Data", was analyzed and modeled using Python with ARIMA (Auto-Regressive Integrated Moving Average) as the primary forecasting method.

Methodology

The project began with data preparation, where global COVID-19 data was filtered to include only information related to Poland. Relevant columns, such as date, new_cases, and total_cases, were retained for the analysis, while missing values were replaced with zeros to ensure compatibility with the ARIMA model. Dates were converted into a suitable format for time-series processing.

Exploratory analysis was conducted to understand trends in the dataset. Visualizations highlighted patterns in total and new cases over time, with a particular focus on the most recent two months to capture short-term trends effectively.

For forecasting, the ARIMA model was applied with parameters (p=3, d=1, q=1) estimated from ACF and PACF plots. A 31-day forecast for December was generated, including confidence intervals. The forecast was visualized alongside historical data to provide a clear view of predicted trends.

Observations

The forecast suggests daily trends for new COVID-19 cases in December, supported by confidence intervals that account for uncertainty. While the predictions provide a useful overview, the accuracy may be limited due to certain shortcomings, such as the replacement of missing data with zeros and the manual selection of ARIMA parameters.

Future enhancements could include improved handling of missing data through interpolation, formal model validation using metrics like RMSE, and exploring alternative forecasting techniques. Despite these limitations, the project lays a foundation for further analysis and offers a preliminary understanding of potential COVID-19 trends in Poland.
