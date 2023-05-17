# Project 2 Zillow

## Project Description

* Zillow is a real estate website located in the U.S. It provides users a quality resource for finding all the details on properties for the purpose of linking customers and realtors for sales. One of the biggest challenges in the real estate industry is predicting the value of a property. The Zillow Project focuses on creating a model that can best predict the current value ('taxvaluedollarcnt') of the property. The p

## Project Goals

* Construct an ML Regression model that predicts propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.

* Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.

* Make recommendations on what works or doesn't work in predicting these homes' values.

* Other key drivers:
    * Does lotsize_sf correlate with taxvalue?
    * Does finished_sf correlate with taxvalue?
    * Is there significance in the means for bedroomcnt and taxvalue?¶
    * Is there significance in the means for garagecarcnt and taxvalue?
## Initial Thoughts

* There are some key indicators in the data that may predict the 'taxvalue' and that those indicators will be evident by the conclusion of the project.

## The Plan

* Acquire the Zillow properties_2017 dataset, joining predictions_2017 with only single family homes with a transaction in 2017 using SQL through a coded acquire .py file.

* Prepare the data using the following columns:
    * target: 'taxvaluedollarcnt' renamed 'taxvalue'
    * features:
        * bedroomcnt 
        * bathroomcnt
        * calculatedfinishedsquarefeet renamed 'finished_sf'
        * garagecarcnt
        * lotsizesquarefeet renamed 'lotsize_sf'
        * yearbuilt
        * fips renamed 'county'

* Explore dataset for predictors of property value ('taxvalue')
    * Answer the following questions:
    * Does lotsize_sf correlate with taxvalue?
    * Does finished_sf correlate with taxvalue?
    * Is there significance in the means for bedroomcnt and taxvalue?¶
    * Is there significance in the means for garagecarcnt and taxvalue?

* Develop a model
    * Using the selected data features develop appropriate predictive models
    * Evaluate the models in action using train and validate splits as well as scaled data
    * Choose the most accurate model 
    * Evaluate the most accurate model using the final test data set
    * Draw conclusions

## Data Dictionary

| Features     | Definition                                                                                                           | Unit        |
|--------------|----------------------------------------------------------------------------------------------------------------------|-------------|
| taxvalue     | Assessed tax value of the home.                                                                                      | US Dollar   |
| bedroomcnt   | Number of bedrooms in home.                                                                                          |             |
| bathroomcnt  | Number of bathrooms in home including half baths.                                                                    |             |
| finished_sf  | Total of square feet in the finished home.                                                                           | Square Feet |
| garagecarcnt | Number of car spots in garage.                                                                                       |             |
| lotsize_sf   | Total of square feet for the property or lot.                                                                        | Square Feet |
| yearbuilt    | Year the home's build.                                                                                               |             |
| county       | County in which the property resides. a.k.a 'fips' 6037=Los Angeles County, 6059=Orange County,  6111=Ventura County |             |

## Steps to Reproduce
1) Clone the repo git@github.com:Brian-ONeil/project1_telco.git in terminal
2) Use personal env.py to connect to download SQL telco dataset
3) Run notebook

## Takeaways and Conclusions

## Recommendations

