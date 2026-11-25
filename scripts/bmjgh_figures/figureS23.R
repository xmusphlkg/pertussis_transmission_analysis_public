source("scripts/bmjgh_figures/supplement_style.R")

countries <- figure_countries
country_axis <- function(x) factor(gsub("_"," ",x),levels=rev(gsub("_"," ",countries)))
factorial <- read.csv("source_data/figures/figureS23_A.csv")
prices <- read.csv("source_data/figures/figureS23_B.csv")
prices$no_expansion_at_nonnegative_price <- tolower(as.character(prices$no_expansion_at_nonnegative_price))=="true"
factorial$country_label <- country_axis(factorial$country)
prices$country_label <- country_axis(prices$country)
factorial$condition <- factor(factorial$burden,
  levels=c("reported_uniform","reported_differentiated","symptomatic_uniform","symptomatic_differentiated"),
  labels=c("Reported\nsame weight","Reported\nseparate weights","Symptomatic\nsame weight","Symptomatic\nseparate weights"))
factorial$direction <- factor(ifelse(factorial$best_expansion_nhb>0,"Higher NHB","Lower NHB"),
                             levels=c("Lower NHB","Higher NHB"))
factorial$label <- formatC(factorial$best_expansion_nhb_per_100k,format="f",digits=2)
stopifnot(nrow(factorial)==32,all(is.finite(factorial$best_expansion_nhb_per_100k)))
p4a <- ggplot(factorial,aes(condition,country_label,fill=direction)) +
  geom_tile(colour="white",linewidth=.6,width=.98,height=.97) +
  geom_text(aes(label=label),size=7/.pt,colour="#111111") +
  scale_y_discrete(limits=rev(gsub("_"," ",countries)),drop=FALSE) +
  scale_fill_manual(values=c("#EADFD5","#B6D8EB"),name="Best-expansion NHB\nper 100,000 people") +
  labs(x=NULL,y=NULL) +
  theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text.x=element_text(size=6))
eligible <- prices[!prices$no_expansion_at_nonnegative_price,]
ineligible <- prices[prices$no_expansion_at_nonnegative_price,]
stopifnot(all(eligible$maximum_expansion_price_multiplier>=0))
p4b <- ggplot(prices,aes(y=country_label)) +
  geom_vline(aes(xintercept=1,linetype="Current price input"),colour="#777777",linewidth=.35) +
  geom_segment(data=eligible,aes(x=0,xend=maximum_expansion_price_multiplier,yend=country_label),
               colour="#0072B2",linewidth=.4) +
  geom_point(data=eligible,aes(x=maximum_expansion_price_multiplier),colour="#0072B2",size=1.8) +
  geom_text(data=ineligible,aes(x=.025,label="None at zero"),hjust=0,size=7/.pt,colour="#555555") +
  scale_y_discrete(limits=rev(gsub("_"," ",countries)),drop=FALSE) +
  scale_linetype_manual(values=c("Current price input"="dashed"),name=NULL) +
  scale_x_continuous(limits=c(-.02,1.08),breaks=c(0,.5,1),labels=c("0","0.5","1.0")) +
  labs(x="Maximum price-input multiplier",y=NULL) +
  theme(axis.text.y=element_blank(),axis.ticks.y=element_blank())
# Check positions after layer-specific training, including both subset layers.
p4built <- ggplot_build(p4b)
stopifnot(identical(as.numeric(p4built$data[[2]]$y),as.numeric(eligible$country_label)),
          identical(as.numeric(p4built$data[[4]]$y),as.numeric(ineligible$country_label)))
figure4 <- p4a+p4b+plot_layout(widths=c(1.65,1),guides="collect")+
  plot_annotation(tag_levels="a") & theme(legend.position="bottom")

export_supplement(figure4, "S23", 183, 108)
