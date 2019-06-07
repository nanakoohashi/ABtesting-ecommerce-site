# Analyze A/B Test Results - README

## Summary
This project aimed to help the company understand if they should implement this new page, keep the old page, or perhaps run the experiment longer to make their decision.


## Steps
1. Wrangle Data.

2. In Part II, we assumed the old page was more effective or equally as effective at converting users than the new page (null hypothesis). 

3. In Part III, we used the logistic regression model to calculate the p-value. 

We also introducted another factor into the regression model, the country in which a user lives. This was to avoid Simpson's Paradox and make sure there was consistency among test subjects in the control and experiment group. 


## Main Findings
- In Part II, we assumed the old page was more effective or equally as effective at converting users than the new page (null hypothesis). This study concluded that with a p-value of 0.906, we could not reject the null hypothesis. We also calculated the z-value which also failed to reject the null hypothesis.

- In Part III, we used the logistic regression model to calculate the p-value. The results also could not reject the null hypothesis. There was no indication that the user's country of residence significantly effected their conversion rate.

We can therefore also conclude that there is no practical reason to adopt the new page as there is no evidence that the new page raises the conversion rates for users and therefore, it is not worth the cost nor time of launching the new page.

