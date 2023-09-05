# libraries
library(ggplot2)
library(dplyr)
library(lubridate)
library(depmixS4)
library(caTools)

# seed: 4747
set.seed(4747)
setwd("C:/Users/brightwu/Courses/CMPT 318/Term Project")

# install.packages('remotes')
# remotes::install_github('vqv/ggbiplot')

# STAGE 1: Feature Engineering
# ------------------------------------------------------------->
# read data from text
df = read.table("TermProjectData.txt", header = TRUE, sep = ",")
df <- na.omit(df)

# scale entire data set after excluding date and time columns
df <- scale(df[c(3:9)], center = TRUE, scale = TRUE)

# pca process and summary
pca <- prcomp(df, center = TRUE, scale = TRUE)
print(pca)
summary(pca)

# plot first two pricinple components
plot(pca$x[,1], pca$x[,2])

# calculate variation percentage of each pca
pca_var <- pca$sdev^2
pca_var_per <- round(pca_var/sum(pca_var)*100, 1)

# scree plot to show pca variation percentage distribution
barplot(pca_var_per, main = "Scree Plot",
      xlab = "Principal Component", ylab = "Percent Variation",
      names.arg = c('PC1','PC2','PC3','PC4','PC5','PC6','PC7'))

pca_data <- data.frame(time = rownames(pca$x), X = pca$x[,1], Y = pca$x[,2])

# plot the graph
ggplot(data = pca_data, aes(x = X, y = Y, label = time)) +
        geom_text() +
        xlab(paste("PC1 - ", pca_var_per[1], "%", sep = "")) +
        ylab(paste("PC2 - ", pca_var_per[2], "%", sep = "")) +
        theme_bw() +
        ggtitle("PCA Graph")

# calculate loading scores
# get two attributes with the greatest loading scores
loading_scores <- pca$rotation[,1]
scores <- abs(loading_scores)
scores_ranked <- sort(scores, decreasing=TRUE)
top_2_attributes <- names(scores_ranked[1:2])
print(top_2_attributes)


# STAGE 2: HMM Training and Testing
# ------------------------------------------------------------->
# read data from text
data_set = read.table("TermProjectData.txt", header = TRUE, sep = ",")

# scale the attributes global_intensity and global_active_power in the table
data_set$Global_intensity = scale(data_set$Global_intensity)
data_set$Global_active_power = scale(data_set$Global_active_power)

# choose time window(7:00AM - 9:00AM) on Tuesday
data_set$Date = as.Date(data_set$Date, "%d/%m/%Y")
data_set$weekday <- weekdays(data_set$Date)
data_set$Time = paste(data_set$Date, data_set$Time)
data_set$Time = as.POSIXlt(strptime(data_set$Time, " %Y-%m-%d %H:%M:%S"))
data_set = filter(data_set, data_set$weekday=='Tuesday' & 
                    hour(data_set$Time)>=7 & 
                    hour(data_set$Time)<10)

# split data set into training set(year 1-2) and test set(year 3)
training_set = subset(data_set, year(data_set$Time)==2006 | 
                        year(data_set$Time)==2007| 
                        year(data_set$Time)==2008)
testing_set = subset(data_set, year(data_set$Time)==2009)

# calculate series of HMMs Loglik and BIC on training_set 
nt <- rep(c(180), times=107)
logLiks <- c()
BICs <- c()
for (n in seq(from = 4, to = 16, by = 2)) {
  model <- depmix(response = list(Global_intensity ~ 1, Global_active_power ~ 1),
                  data = training_set,
                  family = list(gaussian(),gaussian()),
                  nstates = n,
                  ntimes = nt)
  fitModel <- fit(model)
  logLiks = c(logLiks, logLik(fitModel))
  BICs = c(BICs, BIC(fitModel))
  print(n)
}

# Create graphs to find where BIC hits the knee
Ns = seq(from = 4, to = 16, by = 2)
sampleLogLiks <- data.frame(x=Ns, y=logLiks)
sampleBICs <- data.frame(x=Ns, y=BICs)
logLiks_model2 <- lm(y~poly(x,2,raw=TRUE), data=sampleLogLiks)
plot(Ns, logLiks)
x_axis <- seq(1, 16, length=16)
lines(x_axis, predict(logLiks_model2, data.frame(x=x_axis)), col='red')
BICs_model2 <- lm(y~poly(x,2,raw=TRUE), data=sampleBICs)
plot(Ns, BICs)
x_axis2 <- seq(1, 16, length=16)

# Add curve of each model to plot
# lines(x_axis2, predict(BICs_model1, data.frame(x=x_axis2)), col='green')
lines(x_axis2, predict(BICs_model2, data.frame(x=x_axis2)), col='red')

# train HMMs near intersection of Loglik and BIC on training_set
# nstates = 15 under the intersection threshold of Loglik and BIC
nt1 <- rep(c(180), times=107)
model_1 <- depmix(response = list(Global_intensity ~ 1, Global_active_power ~ 1),
                        data = training_set,
                        family = list(gaussian(),gaussian()),
                        nstates = 15,
                        ntimes = nt1)
fitModel_1 <- fit(model_1)
print(logLik(fitModel_1))
print(BIC(fitModel_1))

# nstates = 16 over the intersection threshold of Loglik and BIC
nt2 <- rep(c(180), times=107)
model_2 <- depmix(response = list(Global_intensity ~ 1, Global_active_power ~ 1),
                data = training_set,
                family = list(gaussian(),gaussian()),
                nstates = 16,
                ntimes = nt2)
fitModel_2 <- fit(model_2)
print(logLik(fitModel_2))
print(BIC(fitModel_2))

# construct shell HMM on testing dataset
nt3 <- rep(c(180), times=48)
model_3 <- depmix(response = list(Global_intensity ~ 1, Global_active_power ~ 1),
                  data = testing_set,
                  family = list(gaussian(),gaussian()),
                  nstates = 15,
                  ntimes = nt3)

# apply optimal model parameters
model_3 <- setpars(model_3, getpars(fitModel_1))

# compare normalized training set Loglik/BIC and testing set Loglik/BIC
print("Normalized Training Set Log Likelihood:")
training_loglik = forwardbackward(fitModel_1, return.all=FALSE, useC=TRUE)[3]
normalized_training_loglik = as.numeric(training_loglik) / nrow(training_set)
print(normalized_training_loglik)
print("Normalized Testing Set Log Likelihood:")
testing_loglik = forwardbackward(model_3, return.all=FALSE, useC=TRUE)[3]
normalized_testing_loglik = as.numeric(testing_loglik) / nrow(testing_set)
print(normalized_testing_loglik)


# STAGE 3: Anomaly Detection
# ------------------------------------------------------------->
# read data from text
anomalous_data_set_1 = read.table("DataWithAnomalies1.txt", header = TRUE, sep = ",")
anomalous_data_set_2 = read.table("DataWithAnomalies2.txt", header = TRUE, sep = ",")
anomalous_data_set_3 = read.table("DataWithAnomalies3.txt", header = TRUE, sep = ",")
a_data_sets = list(anomalous_data_set_1, anomalous_data_set_2, anomalous_data_set_3)

for (n in c(1:3)) {
  # scale variables
  a_data_sets[[n]]$Global_intensity = scale(a_data_sets[[n]]$Global_intensity)
  a_data_sets[[n]]$Global_active_power = scale(a_data_sets[[n]]$Global_active_power)
  # choose time window(7:00AM - 9:00AM) on Tuesday
  a_data_sets[[n]]$Date = as.Date(a_data_sets[[n]]$Date, "%d/%m/%Y")
  a_data_sets[[n]]$weekday <- weekdays(a_data_sets[[n]]$Date)
  a_data_sets[[n]]$Time = paste(a_data_sets[[n]]$Date, a_data_sets[[n]]$Time)
  a_data_sets[[n]]$Time = as.POSIXlt(strptime(a_data_sets[[n]]$Time, " %Y-%m-%d %H:%M:%S"))
  a_data_sets[[n]] = filter(a_data_sets[[n]], a_data_sets[[n]]$weekday=='Tuesday' & hour(a_data_sets[[n]]$Time)>=7 & hour(a_data_sets[[n]]$Time)<10)
}

# construct shell HMMs for the three anomalous data sets and transfer parameters
a_hmms = list()
ant <- rep(c(180), times=51)
for (n in c(1:3)) {
  a_hmms[[n]] <- depmix(response = list(Global_intensity ~ 1, Global_active_power ~ 1),
                  data = a_data_sets[[n]],
                  family = list(gaussian(),gaussian()),
                  nstates = 15,
                  ntimes = ant)
  a_hmms[[n]] <- setpars(a_hmms[[n]], getpars(fitModel_1))
  print("Log Likelihood:")
  print(as.numeric(forwardbackward(a_hmms[[n]], return.all=FALSE, useC=TRUE)[3])/nrow(a_data_sets[[n]]))
}

