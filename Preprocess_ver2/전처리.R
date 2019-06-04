library(readxl)
library(xlsx)
library(stringr)


preprocess <- function(filename){
  
  data <- read.xlsx2(filename, 1)
  
  speed = gsub("km/h", "", data$Speed)
  speed = gsub("．",".", speed)
  data$Speed <- as.numeric(speed)
  
  distance = gsub("[A-z]", "", data$Distance)
  distance = gsub("．",".", distance)
  data$Distance <- as.numeric(distance)
  
  for(i in 6:19){
    
    values = gsub("[A-z]","", data[,i])
    values = gsub("．",".", values)
    data[,i] = as.numeric(values)
    
  }
  
  plus_power = rep(0.0, nrow(data))
  
  i = 6
  
  while(i <= 14){
    
    power = data[,i]*data[,i+1]
    plus_power = plus_power + power/1000
    i = i + 2
    
  }
  
  minus_power = rep(0, nrow(data))
  
  i = 16
  
  while(i <= 18){
    
    power = data[,i]*data[,i+1]
    minus_power = minus_power + power/1000
    i = i + 2
    
  }
  
  plus_power = plus_power + minus_power
  
  total_power = rep(0, nrow(data))
  
  for(j in 1:nrow(data)){
    if(data[j,9] == 1){
      total_power[j] =  - plus_power[j]
    }else{
      total_power[j] =  plus_power[j]
    }
  }
  
  
  new_df = data.frame(data$No, data$Time, data$Speed, data$Next.Station.code, data$Distance, data$Car2_VVVF.POWERING, data$Car2_VVVF.BRAKING, total_power)
  colnames(new_df) = c("No","Time","Speed","Next.Station.code","Distance","Powering","Braking","Power")
  
  sub_df = subset(new_df, new_df$Next.Station.code != "노포(134)" &  new_df$Next.Station.code != "다대포 해수욕장(95)"  &  new_df$Next.Station.code != "범어사(133)"
                  & new_df$Next.Station.code != "x(0)")
  return(sub_df)
  
}

sub_dir = c("2018년 7월 19일","2018년 8월 14일","2018년 9월 4일","2018년 9월 21일")
train_num = c("49편성","50편성", "51편성")


for(date in sub_dir){
    for(num in train_num){
      dir = sprintf("C:/Users/jbk48/Desktop/민섭/철도연/download_webdisk.krri.re.kr(2019-04-14)/%s/%s", date, num)
      setwd(dir)
      
      filenames = readLines("list.txt")
      sub_file = str_sub(filenames, end = 11)
      df = data.frame(filenames, sub_file)
      split_df = split(df, df$sub_file)
      
      dir.create("./전처리")
      
      i = 1
      
      for(sub_df in split_df){
        sub_df = data.frame(sub_df)
        final_df = data.frame()
        for (filename in as.character(sub_df[,1])){
          final_df = rbind(final_df,preprocess(filename))
        }
        write.csv(final_df,sprintf("./전처리/data_%d.csv",i), sep ="," , row.names = FALSE)
        i = i + 1
      }
    }
}
