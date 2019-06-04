library(stringr)
library(tidyr)
library(readr)


setwd("C:/Users/jbk48/Desktop/민섭/철도연/download_webdisk.krri.re.kr(2019-04-14)/2018년 9월 21일/51편성")

dir.create("./전처리2")


for(j in 1: length(dir("./전처리"))){
  
  data = read.csv(sprintf("./전처리/data_%d.csv",j) , sep =",")
  
  if(nrow(data) > 0){
    
    data$Next.Station.code = as.character(data$Next.Station.code)
    
    code_number = parse_number(data$Next.Station.code)
    
    rle_table = rle(code_number)
    index = rle_table$lengths
    values = rle_table$values
    
    if(length(values)==1){
      next
    }
    
    if(values[1] < values[2]){
      head_north_index = c(1)
    }else{
      head_north_index = c(0)
    }
    
    i = 2
    while(i <= length(values)){
      
      if(values[i] > values[i-1]){
        head_north_index = append(head_north_index, 1)
      }
      else{
        head_north_index = append(head_north_index, 0)
      }
      i = i + 1
      
    }
    
    head_north = rep(head_north_index, times = index)
    final_df = cbind(data, head_north)
    final_df = subset(final_df, final_df$Next.Station.code != "범어사(133)" & final_df$Next.Station.code != "다대포항(96)")
    
    
    write.csv(final_df, sprintf("./전처리2/data_%d.csv",j) , sep="," , row.names = FALSE)
  }
  
}