clear
import delimited "/Users/arturbegyan/Desktop/Submission/operating_dataset.csv", encoding(ISO-8859-1)
set more off

gen t = _n
tsset t

* In-line nowcasting experiments and F-tests
reg consumption L.consumption tbill L.income smp500 v132 v91 v105 v166 v110 v9 v69 v6 v24 v86 v35 v150 v29 v127 v45 v40 v50 v15 v81 v60 v140 v152 v19 v44 v8
test v132 v91 v105 v166 v110 v9 v69 v6 v24 v86 v35 v150 v29 v127 v45 v40 v50 v15 v81 v60 v140 v152 v19 v44 v8

reg consumption L.consumption tbill L.income smp500 michigan_index
test michigan_index

reg consumption L.consumption tbill L.income smp500 cci
test cci

* Consumption ARMA(p,q) - information criteria

arima consumption, arima (1,0,0) 
estat ic

arima consumption, arima (2,0,0) 
estat ic

arima consumption, arima (0,0,1) 
estat ic

arima consumption, arima (0,0,2) 
estat ic

arima consumption, arima (1,0,1) 
estat ic

arima consumption, arima (1,0,2) 
estat ic

arima consumption, arima (2,0,1) 
estat ic

arima consumption, arima (2,0,2) 
estat ic

* ACF and PACF plots of consumption yoy
pac consumption
ac consumption

* Tests for serial correlation in residuals
reg consumption L.consumption
estat bgodfrey, lags(1/4)

* ADF tests for non-GT variables

dfuller consumption, constant regress lags(1)
dfuller income, constant regress lags(1)
dfuller smp500, constant regress lags(1)
dfuller tbill, constant regress
dfuller cci, constant regress lags(3)
dfuller michigan_index, constant regress

* Produce summary statistics for non-GT variables
su consumption
su income
su smp500
su tbill
su cci
su michigan_index



