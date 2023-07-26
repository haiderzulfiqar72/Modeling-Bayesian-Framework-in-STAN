---
title: "Bayesian Inference To Estimate Difference in Arrest Rates Based on Domestic and Non Domestic Crimes"
author: "Haider Zulfiqar"
date: "December 22, 2022"
output:
  pdf_document: default
---
# Describe the data, and document the origin of it  
Dataset used in the following work originates from the Chicago Police Department. It reflects the crime incidents (with the exception of murders) that occurred in the City of Chicago from 2001 till 2020. Dataset has 22 feature variables explaining various details for criminal activities. It can be found here: https://data.cityofchicago.org/Public-Safety/Chicago-Police-Department-Illinois-Uniform-Crime-R/c7ck-438e

Our focus of interest for this particular work would revolve around examining two particular column features, 'Arrest' and 'Domestic'. 'Arrest' indicates whether an arrest was made while 'Domestic' indicates whether the incident was domestic-related as defined by the Illinois Domestic Violence Act. And we will derive inference if the domestic violence contributed more towards the arrest rate or the non domestic violence attributed more.

```{r warning=FALSE, message=FALSE, results='hide'} 
library(data.table)
library(rstan)
library(bayesplot)
```
# Loading the data
```{r}
dataset<- fread("Crimes_-_2001_to_Present.csv")
df <- dataset[,.(.N, y=sum(Arrest)),.(Domestic)]
df |> head()
```  
As evident from the above data, difference between the total number arrests when there was a non domestic crime compared with when there was a domestic crime looks quite significant and a particular case of interest. Thus, our model will try to generate inference based on the above and try to answer if that is the case.
  
# Statistical basis of the method applied to the selected data.
As such, we will apply Bayesian Inference to draw conclusion about the difference in the two cases. Regarding how our data is distributed, we can quickly infer binomial distribution to be the case with domestic and non domestic crimes as there are only two possible scenarios here. We will derive posterior distribution for our model to draw comparison between the two and any overlap would showcase similarities in the pattern and vice versa.

To infer if the distribution is the same or not, a STAN model would be applied. Stan uses gradient-based MCMC to perform Hamiltonian Monte Carlo to get approximate simulations from the posterior distribution. Similar to BUGS and JAGS, it allows a user to write a Bayesian model in a user friendly language whose code looks like statistics notation; and sample from the posterior distribution based on the ratio of the relative density functions of the two distributions in question. 

So the goal of sampling here is to draw from a density p($\theta$) for parameters $\theta$. This is typically a Bayesian posterior p($\theta$|y) given data y, and in particular, a Bayesian posterior coded as a Stan program. As for model limitation goes, generally inference and optimization can never be performed adequately to the full capacity; there will always be optimization or inference problems that are beyond our control. It is the goal of Stan to push these capacities to derive maximum convergence for the model as will be shown in the report. 

In our case, exploiting the concept of conjugate priors, beta distribution has been set as the prior distribution for both p_dom (Domestic) and p_nodorm (Non Domestic). 

# Stan Model
```{r}
model = "

data {
int n[2];
int y[2];
}

parameters {
real<lower=0> p_dom;
real<lower=0> p_nodom;
}

model {
// Likelihood
y[1] ~ binomial(n[1], p_nodom);
y[2] ~ binomial(n[2], p_dom);
// prior
p_dom ~ beta(1,1);
p_nodom ~ beta(1,1);
}
"
```  
Our model samples from binomial distribution for both the parameters p_nodom (No Domestic) and p_dom (Domestic) with the same non informative Beta Priors to uphold conjugacy. 
  
# Deriving Bayesain Inference, Model Evaluation and Results 
```{r}
stan.model <- stan(model_code=model,
                      data=list(n=df[,N], y=df[,y]),
                      chains = 4,
                      warmup = 500,
                      iter = 1000,
                      cores = 4, 
                      thin = 1
)

mcmc_dens(stan.model)
posterior <- as.matrix(stan.model)
summary(posterior)
mcmc_areas(posterior, pars = c("p_dom"),prob=0.95)
mcmc_areas(posterior, pars = c("p_nodom"),prob=0.95)
plot(stan.model, ci_level = 0.95, outer_level = 0.999)
```

As evident from our model, there is a significant difference between p_dom and p_nodom values. The model shows, 95% Confidence Interval for values in the vicinity of 0.1945 and 0.1965 for p_dom while it is between 0.2847 and 0.2853 for p_nodom. Hence we can draw conclusion from the above inference that there is a viable difference between arrests owing to non-domestic crimes compared with arrests as a result of domestic crimes. 

# Model Validation
```{r}
print(stan.model, pars=c("p_dom", "p_nodom"))
mcmc_trace(stan.model)
```

To validate results from the above model, the analysis from Stan model also reports an effective sample size and how well the chains are converging / mixing (RË close to 1 indicate good mixing as is the case here) and it can be good measure of how reliable our model and results are. Trace plots of the fit further validates our analysis and results that all four chains overlaid for each parameter i.e. the chains appear to be mixing well for both the parameters.
