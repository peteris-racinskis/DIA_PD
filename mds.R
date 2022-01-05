dist.de <- read.table("Results-1980-2015-dist.txt")
names <- read.table("Results-1980-2015-name.txt")
#dist.de <- read.table("classifier-distances.txt")
#names <- read.table("classifier-names.txt")
fit <- cmdscale(dist.de, eig = TRUE, k = 2)
x <- fit$points[, 1]
y <- fit$points[, 2]
#plot(x,y, pch=4, col="grey",xlim=c(-1.5,-0.8))
plot(x,y, pch=4, col="grey")
text(x,y,labels=names$V1)

library(tsne)
plot.tsne <- function(df, lab=None, perp=5, iter=400, class=NULL, plt=T, k=2, ret=F) {
  transformed <- tsne(df, k = k, perplexity = perp,
                      initial_dims = length(df[1,]), max_iter = iter)
  x <- transformed[,1]
  y <- transformed[,2]
  if (plt) {
    plot(x,y,col="grey",lxlab="comp1",
         ylab="comp2",
         main=sprintf("TSNE", distance))
    text(x,y,labels=lab)
  }
  if (ret) {
    transformed
  }
}
t <- tsne(as.dist(dist.de))
x <- t[,1]
y <- t[,2]
plot(x,y, pch=4, col="grey")
text(x,y,labels=names$V1)