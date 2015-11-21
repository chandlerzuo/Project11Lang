## Function

library(gbm)
library(data.table)
library(xgboost)
library(randomForest)
library(doParallel)
library(ggplot2)
library(stringr)


#' @name Add polynomial features to the data set.
#' @details 
#' @export
AddPolyFeatures <- function(dat) {
  return(list(data = cbind(dat,(dat$Month)^2,(dat$Week)^2,(dat$Day)^2)))
}


#' @name Add triangluar features to the data set.
#' @details Try to fit a triangular function on the month, week, day variables as their nonlinear patterns with the response variable.
#' If 'fit=TRUE', then we try to find triangular functions of the month, week and day variables according to their pattern with the response variable.\cr
#' If 'fit=FALSE', we take the coefficients from 'month.tri.coef', 'week.tri.coef' and 'day.tri.coef' to calculate the triangular features, and add them to the original data set.
#' @export
AddTriFeatures <- function(dat, response, fit = FALSE, month.tri.coef = NULL, day.tri.coef = NULL, week.tri.coef = NULL) {

    MonthTriObj <- function(x) {
        return(sum((response - sin((dat$Month / x[1] - x[2]) * 180 * pi)) ^ 2))
    }

    DayTriObj <- function(x) {
        return(sum((response - sin((dat$Month / x[1] - x[2]) * 180 * pi)) ^ 2))
    }

    WeekTriObj <- function(x) {
        return(sum((response - sin((dat$Day / x[1] - x[2]) * 180 * pi)) ^ 2))
    }

    if(fit) {
        month.tri.coef <- optim(c(12, 0), MonthTriObj, method = 'L-BFGS-B')$par
        day.tri.coef <- optim(c(365, 0), MonthTriObj, method = 'L-BFGS-B')$par
        week.tri.coef <- optim(c(6, 0), MonthTriObj, method = 'L-BFGS-B')$par
    }

    return(list(data = cbind(dat,
                             sin(dat$Month / month.tri.coef[1] - month.tri.coef[2]),
                             sin(dat$Day / day.tri.coef[1] - day.tri.coef[2]),
                             sin(dat$Week / week.tri.coef[1] - week.tri.coef[2])),
                month.tri.coef = month.tri.coef,
                day.tri.coef = day.tri.coef,
                week.tri.coef = week.tri.coef))
}

#' @name Construct regressor matrix.
#' @details Eliminate the columns with no variance and non-numeric columns.
#' Columns in 'omit' and non-numeric columns are omitted.
#' @param omit Columns to omit.
#' @return A list of two variables. 'regressors' is the list of column names from the original data matrix. 'xreg' is the data matrix to be used in modeling.
#' @export
ConstructRegressors <- function(dat, omit = c(), regressors = c()) {
    regressors <- NULL
    xreg <- NULL
    for(col in names(dat)) {
        if(col %in% omit) {
            next
        }
        if(!class(dat[[col]]) %in% c('numeric', 'integer')) {
            next
        } else {
            if(var(dat[[col]]) > 0) {
                xreg <- cbind(xreg, dat[[col]])
                regressors <- c(regressors, col)
            }
        }
    }
    colnames(xreg) <- paste("V", seq(ncol(xreg)), sep = "")
    return(list(regressors = regressors, xreg = xreg))
}

#' @name Fit the machine learning model based on the training data set.
#' @param dat The training data set.
#' @param log.transform Whether we log-transform the response variable.
#' @param add.tri Whether we add triangular functions as predictors.
#' @param ntress Number of trees in the model.
#' @param maxdepth The maximum tree depth.
#' @param method Method to be used: 'gbm' or 'rf'.
#' @return
#' @export
model.train.mlts <- function(dat, log.transform, add.tri, add.poly, ntrees, maxdepth, method) {
    dat <- dat[dat$Open == 1, ]
  
    response <- NULL
    if(log.transform) {
        response <- log(dat$Sales + 1)
    } else {
        response <- dat$Sales
    }
    message('transformed')
    
    month.tri.coef <- week.tri.coef <- day.tri.coef <- NULL
    if(add.tri) {
        add.tri.info <- AddTriFeatures(dat, response, fit = TRUE)
        dat <- add.tri.info$data
        month.tri.coef <- add.tri.info$month.tri.coef
        week.tri.coef <- add.tri.info$week.tri.coef
        day.tri.coef <- add.tri.info$day.tri.coef
    }
    message('added triangular variables')

    if(add.poly) {
      add.poly.info <- AddPolyFeatures(dat)
      dat <- add.poly.info$data
    }
    message('added polynomial variables')

    ret <- ConstructRegressors(dat, c('Sales', 'Customers', 'Date'))
    modeldata <- data.frame(cbind(response, ret$xreg))
    names(modeldata)[1] <- 'Response'

    message('constructed fit matrix')

    if(method == 'gbm') {
      fit <- gbm(Response ~ ., data = modeldata, distribution="gaussian", n.trees = ntrees, interaction.depth = 8)
    } else {
      tun <- tuneRF(modeldata[,!(names(modeldata) %in% c("Response"))], modeldata$Response, stepFactor=1,plot=FALSE)
      fit <- randomForest(Response ~ ., data = modeldata, ntrees = ntrees, mtry=print(tun)[1,1], maxnodes = 2 ^ maxdepth)
    }

    return(list(fit = fit, regressors = ret$regressors,
                month.tri.coef = month.tri.coef,
                week.tri.coef = week.tri.coef,
                day.tri.coef = day.tri.coef))
}

#' @name Predict on the tuning data set.
#' @param dat The tuning data set.
#' @param model The result returned from 'model.train.mlts' function.
#' @param log.transform Whether the response is log-transformed.
#' @param add.tri Whether triangular signals are added as predictors.
#' @export
model.predict.mlts <- function(dat, model, log.transform, add.tri, add.poly) {
    ## Only fit for open days
    dat1 <- dat[dat$Open == 1, ]

    if(add.tri) {
        dat1 <- AddTriFeatures(dat1, response = NULL,
                               fit = FALSE,
                               month.tri.coef = model$month.tri.coef,
                               week.tri.coef = model$week.tri.coef,
                               day.tri.coef = model$day.tri.coef
                               )$data
    }
    
    if(add.poly) {
      add1.poly.info <- AddPolyFeatures(dat1)
      dat1 <- add1.poly.info$data
    }
    
    
    xreg <- NULL
    for(col in model$regressors) {
        xreg <- cbind(xreg, dat1[[col]])
    }
    colnames(xreg) <- paste("V", seq(ncol(xreg)), sep = "")
    predicted <- predict(model$fit, data.frame(xreg), n.trees = model$fit$n.trees)

    ## Log transformation
    if(log.transform) {
        predicted <- exp(predicted) - 1
        predicted[predicted < 0] <- 0
    }

    ## Final prediction
    dat[dat$Open == 1, "Predicted"] <- predicted
    dat[dat$Open == 0, "Predicted"] <- 0
    return(dat$Predicted)
}

#' @details This function will iterate across all possible options of the 'model.train.mlts' function to train model based on the training data set, and pick the final model based on the best prediction on the tuning data set.
#' @export
FitAndPredict <- function(dat.fit, dat.tune) {

    pars <- c()
    k <- 1
    for(log in c(TRUE, FALSE)) {
        for(add.tri in c(TRUE, FALSE)) {
          for(add.poly in c(TRUE, FALSE)){
            for(method in c('rf', 'gbm')) {
              for(maxdepth in c(8,16,32)) {
                for(ntrees in c(1e3, 2e3,5e3,1e4)) {
                  pars[[k]] <- c(log, add.tri, add.poly, method, maxdepth, ntrees)
                  k <- k + 1
                }
              }
            }
          }
        }
    }

    results <- list()
    #k=1
    for(k in seq(length(pars))) {
        model.result <- model.train.mlts(dat.fit,
                                         log.transform = as.logical(pars[[k]][1]),
                                         add.tri = as.logical(pars[[k]][2]),
                                         add.poly = as.logical(pars[[k]][3]),
                                         method = pars[[k]][4],
                                         ntrees = as.integer(pars[[k]][6]),
                                         maxdepth = as.integer(pars[[k]][5])
                                         )
        
        model.predict <- model.predict.mlts(dat.tune,
                                            model.result,
                                            log.transform = as.logical(pars[[k]][1]),
                                            add.tri = as.logical(pars[[k]][2]),
                                            add.poly = as.logical(pars[[k]][3])
                                            )

        results[[k]] <- list(mse = mean((model.predict - dat.tune$Sales) ^ 2),
             predict = model.predict)
    }

    best.loss <- Inf
    best.predict <- NULL
    best.k <- c()
    for(k in seq_along(results)) {
        if(results[[k]]$mse < best.loss) {
            best.loss <- results[[k]]$mse
            best.predict <- results[[k]]$predict
            best.k <- k
        }
    }

    return(list(predict = best.predict,
                mse = best.loss,
                log.transform = as.logical(pars[[best.k]][1]),
                add.tri = as.logical(pars[[best.k]][2]),
                add.poly = as.logical(pars[[best.k]][3]),
                method = pars[[best.k]][4],
                ntrees = as.integer(pars[[best.k]][6]),
                maxdepth = as.integer(pars[[best.k]][5])
                ))
}

