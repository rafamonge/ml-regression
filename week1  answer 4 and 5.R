### 4

w0 = 4569
w1 = 143

p <- function(x){
  w0 + w1 * x
}

FourthResult <- p(10) / w1


###  5
w0_51 <- -44850
w1_51 <- 280.76

conversion_squareFeetToMeters <- 0.092903

w0_52 <- -44850 
w1_52 <- 280.76 * (1 / conversion_squareFeetToMeters)

squareFeet <- function(x){
  w0_51 + w1_51 * x
}

xSquare <- 1000
xMeter <- conversion_squareFeetToMeters * xSquare

squareMeters <- function(x){
  w0_51 + w1_52 * x
}
Answer5 <-  w1_52
  
  