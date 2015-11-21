## Main
rm(list = ls())
gc()


dir <- "/Users/chengmingwang/Dropbox/Kaggle/Sales"
codedir <- paste(dir,"codes/ChW",sep="/")
datadir <- paste(dir,"data",sep="/")
outputdir <- paste(dir,"output",sep="/")

n.cores <- 3

load(file.path(datadir, 'pretreat/data.Rda'))

## Test a single function
if(FALSE) {
    dat <- numeric(train.fulldata)[Store == 1,]
    dat.fit <- dat[Date < '2015-1-1', ]
    dat.tune <- dat[Date >= '2015-1-1', ]

    model.result <- model.train.mlts(dat.fit, method = 'rf')
    model.predict <- model.predict.mlts(dat.tune, model.result)
    plot(dat.tune$Sales, model.predict)
}

if(FALSE) {
    if(n.cores > detectCores()) {
        if(detectCores() > 1) {
            n.cores <- detectCores() - 1
        } else {
            n.cores <- 1
        }
    }
    cl <- registerDoParallel(n.cores)
}





##foreach(i = seq(10)) %dopar% {
#i <- 1
for(i in seq(10)) {
    dat <- numeric(train.fulldata)[Store == i,]
    dat.fit <- dat[Date < '2015-1-1', ]
    dat.tune <- dat[Date >= '2015-1-1', ]

    result <- FitAndPredict(dat.fit, dat.tune)

    pdf(file.path(datadir, paste('../output/ml_ts+tune/tune_store_', i, '.pdf', sep = "")))
    print(ggplot() + geom_point(aes(x = dat.tune$Sales, y = result$predict)) + xlab("Sales") + ylab(paste("Pred",
                                                                                                          "log=", result$log.transform,
                                                                                                          "tri= ", result$add.tri,
                                                                                                          "poly= ", result$add.poly,
                                                                                                          "meth= ", result$method,
                                                                                                          "ntrees= ",result$ntrees,
                                                                                                          "maxdepth=", result$maxdepth)) + 
            ggtitle(paste("MSE =", round(result$mse, 0))) + geom_abline(intercept = 0, slope = 1))
    dev.off()
}

## stopCluster(cl)
