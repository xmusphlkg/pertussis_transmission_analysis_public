source("scripts/bmjgh_figures/supplement_style.R")

waning <- do.call(rbind,lapply(c(1,3,6,10),function(d) {
  t<-seq(0,20,length.out=201);z<-2*t/d
  data.frame(mean_years=d,time_years=t,protected_fraction=exp(-z)*(1+z))
}))
stopifnot(nrow(waning)==804,all(waning$protected_fraction>=0),all(waning$protected_fraction<=1))
waning$duration <- factor(waning$mean_years,levels=c(1,3,6,10),labels=c('1 year','3 years','6 years','10 years'))
ws <- ggplot(waning,aes(time_years,protected_fraction,colour=duration))+
  geom_line(linewidth=.55)+
  scale_colour_manual(values=c('#747678','#0072B2','#D55E00','#8064A2'),name='Mean protection duration')+
  scale_x_continuous(breaks=c(0,5,10,15,20),expand=expansion(mult=c(.02,.02)))+
  scale_y_continuous(breaks=c(0,.25,.5,.75,1),labels=scales::label_percent(),
                     expand=expansion(mult=c(.02,.03)))+
  labs(x='Time since vaccination (years)',y='Proportion remaining in protected states')+
  theme(legend.position='bottom')

export_supplement(ws, "S2", 183, 90)
