## Chargement des librairies 
library(readr)
library(dplyr)

##  Prétraitement
donnees_nt <- read_csv("Data/game.csv")

#Avant le 3 fevrier 1980, les données sont partiellement corrompues
donnees_propres <- donnees_nt %>% filter(game_date>="1980-02-03" & !is.na(fg3a_home) & season_type != "All-Star" & season_type != "All Star") %>% select(season_id,game_date,fg3a_home,fg3m_home) 
write.csv(donnees_propres, "Data/donnees_propres")

#il y a plusieurs matchs par jour, mais on n'a pas plus de précision
# sur la date que le jour du match (pas l'heure), donc on moyenne 
# tout sur une journée :
donnees_journalieres <- donnees_propres %>% group_by(game_date, season_id) %>% summarise(mean_3pa = mean(fg3a_home)) %>% ungroup()
write.csv(donnees_journalieres,"Data/donnees_journalieres")
View(donnees_journalieres)

## Visualisation des données
library(dygraphs)
library(zoo)
library(xts)
#objet xts nécessaire pour la visualisation dynamique avec dygraph
serie_tempo_3pts = xts(x = donnees_journalieres$mean_3pa,order.by = donnees_journalieres$game_date)
dygraph(serie_tempo_3pts) %>% dyRangeSelector()
#on peut déjà constater l'évolution fulgurante du nombre de 3pts
#tentés par match après la saison 2009-2010, la première de Steph Curry


## Tendance par moyenne mobile
#tendance sur x jours pour éliminer le bruit. moyenne mobile
tendance_180j <- rollmean(x = donnees_journalieres$mean_3pa, k = 180, fill = NA)
tendance_365j <- rollmean(x = donnees_journalieres$mean_3pa, k = 365, fill = NA)
tendance_5a <- rollmean(x = donnees_journalieres$mean_3pa, k = 5*365, fill = NA)

#Pour comparer la courbe à sa tendance supposée, et ainsi vérifier
#la pertinance de la fenêtre de 180 jours choisie, on fait : 
serie_tendance180 <- cbind(serie_tempo_3pts,tendance_180j)

dygraph(serie_tendance180) %>% dyRangeSelector()


serie_tendance365 <- cbind(serie_tempo_3pts,tendance_365j)
dygraph(serie_tendance365) %>% dyRangeSelector()

serie_tendance5a <- cbind(serie_tempo_3pts,tendance_5a)
dygraph(serie_tendance5a) %>% dyRangeSelector()
#On constate que cette méthode est plutôt inadaptée: 
#-  pour des fenêtres de l'ordre d'une saison NBA (~ 1 an), 
# la tendance observée est trop sensible aux petites variation inter-saisons,
# celles-ci n'étant pas toujours significatives pour l'évolution générale.
#-  pour des fenêtres plus larges (5 ans), on perd la courbe de tendance
# sur les premières données et sur les dernières, alors que les
# dernières données semblent indiquer une baisse des 3pts tirés,
#qu'on aimerait voir apparaître sur la courbe de tendance.



## Tendance par GAM
library(nlme)
library(mgcv)

#pour utiliser gam, le temps ne doit pas être au format date mais
# au format numérique. On le convertit ainsi : 
Time = 1:length(donnees_journalieres$game_date)
#On ajoute ensuite ce nouveau tableau d'entiers consécutifs en
#tant que nouvelle colonne dans les données
donnees_journalieres <- mutate(donnees_journalieres,Time)


# Gam avec k = 30 (degrès de liberté max), on ajuste ensuite
modele_gam = gam(mean_3pa~s(Time,k = 30), data = donnees_journalieres)

summary(modele_gam)
#Le summary montre qu'environ 26 degrès de libertés ont été conservés
#soient 1 transition de la courbe toutes les saisons et demi. C'est trop,
#car ça prend surement aussi en compte la saisonnalité, alors qu'on ne
#veut que la tendance globale de la courbe. Traçons : 

tendance_gam_k30 = xts(x = modele_gam$fitted, order.by = donnees_journalieres$game_date)
serie_tendance_gam_k30 = cbind(serie_tempo_3pts,tendance_gam_k30)
dygraph(serie_tendance_gam_k30) %>% dyRangeSelector()
#Comme attendu, le nombre de 26 degrés de liberté est bien trop
#important, on perd l'intérêt d'être passé de la moyenne mobile
# à du GAM, puisqu'on retrouve une courbe en escalier qui subit
# les variations saisonnières. On va forcer un k = 4

# GAM k = 4 :
modele_gam = gam(mean_3pa~s(Time,k = 4), data = donnees_journalieres)
summary(modele_gam)
tendance_gam_k4 = xts(x = modele_gam$fitted, order.by = donnees_journalieres$game_date)
serie_tendance_gam_k4 = cbind(serie_tempo_3pts,tendance_gam_k4)
dygraph(serie_tendance_gam_k4) %>% dyRangeSelector()

# GAM k = 11 :
modele_gam = gam(mean_3pa~s(Time,k = 11), data = donnees_journalieres)
summary(modele_gam)
tendance_gam_k11 = xts(x = modele_gam$fitted, order.by = donnees_journalieres$game_date)
serie_tendance_gam_k11 = cbind(serie_tempo_3pts,tendance_gam_k11)
dygraph(serie_tendance_gam_k11) %>% dyRangeSelector()

# GAM k = 20 :
modele_gam = gam(mean_3pa~s(Time,k = 20), data = donnees_journalieres)
summary(modele_gam)
tendance_gam_k20 = xts(x = modele_gam$fitted, order.by = donnees_journalieres$game_date)
serie_tendance_gam_k20 = cbind(serie_tempo_3pts,tendance_gam_k20)
dygraph(serie_tendance_gam_k20) %>% dyRangeSelector()
#Le choix d'une bonne valeur de k est difficile : d'un côté
#les valeurs failbles (4 ou 5) ignorent la bosse de 1995 à 2000,
# qui est significative sur la période considérée. De l'autre, 
# cette bosse est due à un changement des règles NBA facilitant les
# 3 points sur la période, mais ayant été annulé ensuite.
# Considérer cette bosse comme une tendance est donc peut-être un 
#mauvais choix pour faire des prédictions sur les données. 

#Pour comprendre le problème, traçons la courbe du GCV, qui évalue
#la proximité entre la courbe du modèle GAM et les données, 
# en pénalisant les degrés de liberté, en fonction de k, nbre max
# de deg de liberté :


# Initialisation des valeurs de k à tester (de 3 à 50)
valeurs_k <- 3:50
# Création d'un vecteur vide pour stocker les résultats
resultats_gcv <- numeric(length(valeurs_k))
# Lancement de la boucle
for (i in 1:length(valeurs_k)) {
  k_actuel <- valeurs_k[i]
  modele_temp <- gam(mean_3pa ~ s(Time, k = k_actuel), data = donnees_journalieres)
  
  # On stocke le GCV dans notre vecteur de résultats
  resultats_gcv[i] <- modele_temp$gcv.ubre
}

# Affichage graphique de l'évolution du GCV en fonction de k
plot(valeurs_k, resultats_gcv, type = "b", pch = 20, 
     xlab = "Valeur de k", ylab = "Score GCV", 
     main = "Évolution du GCV selon la flexibilité")


# Résultats : en raison de la forte corrélation entre le nbre de 3pts
#d'aujourd'hui et celui de la veille, le GCV est d'autant plus bas
# que k est grand. Cela ne signifie pas pour autant qu'une bonne
#tendance est donnée par un modele GAM avec un grand k. 
# On va tester la pertinence de toutes ces tendances dans le cadre
# de prédiction, sur un critère de validation croisée par bloc. 


## Validation croisée par bloc
#Idée : on n'arrive pas à savoir quelle valeur de k prendre pour
# le modèle GAM, simplement en regardant les courbes. Solution : 
#on divise les données en 10 blocs de 4 saisons NBA, on en retire 
#un des données et on regarde quelle valeur de k permet au modèle
# GAM de prévoir au mieux le bloc manquant


Nblock <- 10 #Le nombre de block pour la VC
borne_block <- seq(1, nrow(donnees_journalieres), length = Nblock+1) %>% floor
#Liste des 11 indices délimitant les 10 blocs de la VC par bloc

block_list <- list()
for (i in 2:Nblock){
    block_list[[i-1]] <- borne_block[i-1]:(borne_block[i]-1)
}
block_list[[Nblock]] <- borne_block[Nblock]:(borne_block[Nblock+1])
#Liste contenant, en case i, les indices du bloc i des données


block_res <- function(equation, block){
  modele <- gam(equation, data = donnees_journalieres[-block,])
  #Gam sur tous sauf le block en argument : il y a un trou
  
  prevision <- predict(modele,newdata = donnees_journalieres[block,])
  #fonction de R prédisant à partir du modele donné les données
  #du bloc manquant
  
  return(donnees_journalieres$mean_3pa[block] - prevision)
}
#La fonction block_res prend en argument une equation et un bloc
# et applique le modele GAM à donnees_journalieres privée du bloc
#selon l'equation donnée. Elle calcule ensuite l'écart entre données
#et prévision sur ces données privée du bloc. 


#On va maintenant appliquer cette fonction à chaque bloc, calculer 
# l'erreur quadratique pour différente valeur de k, et faire un
#choix final pour la valeur de k

#Itération sur k de 3 à 15 (le GCV ne varie plus après: 
resultat_mse = numeric(13) #liste des erreurs quadratiques moyennes
for(k_actuel in 3:15){ 
  formule <- as.formula(paste0("mean_3pa ~ s(Time,k = ",k_actuel,")"))
  residus_k = lapply(block_list, block_res, equation = formule) %>% unlist()
  #la fonction lapply applique à tous les élément de la liste
  #donnée en argument, la fonction donnée en 2e arg. Elle renvoie 
  #une liste, donc on applique unlist pour avoir un vecteur
  residus_k_2 = mean(residus_k^2)
  resultat_mse[k_actuel-2] <- residus_k_2
}
#on a maintenant un vecteur contenant, à l'indice k-2, 
#l'erreur quadratique moyenne de la CVB sur un modele GAM de 
#flexibilité k. On va trouver le k rendant cette erreur minimale.

min_mse = min(resultat_mse)
k_final <- which.min(resultat_mse) + 2 #réindexation pour obtenir le bon k

#La tendance minimisant le risque quadratique, selon la validatio
#croisée par bloc, est donc : 
modele_final = gam(mean_3pa~s(Time,k = k_final), data = donnees_journalieres)
summary(modele_final)
## Tracé : Données + Tendance
tendance = xts(x = modele_final$fitted, order.by = donnees_journalieres$game_date)
serie_tendance <- cbind(serie_tempo_3pts,tendance)
dygraph(serie_tendance) %>% dyRangeSelector()
#En conclusion, minimiser le risque quadratique demande de limiter
#la flexibilité, quitte à ignoré la bosse de 1995 à 2000, pour
#éviter le sur-apprentissage



## ACF et PACF :
#On va maintenant tracer l'ACF et la PACF pour déterminer si 
#d'autres informations temporelles régulières ne sont pas décrite
# avec la tendance GAM qu'on a choisi

acf(modele_final$residuals)
pacf(modele_final$residuals)


# Ces 2 diagrammes montrent qu'un autre phénomène que la tendance
# intervient dans la dépendance entre les données sur le court
# terme. On s'attend par exemple à ce qu'un joueur qui a beaucoup
# shooté un certain jour, shoot autant les quelques jours suivant
# pour profiter d'un haut taux de réussite passager par exemple. 
# La seconde hypothèse est celle d'une période dans la stratégie
# des équipe. En effet, un bon shooter à 3pts est un atout fort pour
# une équipe. On peut s'attendre à ce que l'entraîneur fasse 
# moins jouer un tel atout dans les semaines précédent les play-off
# et le fasse plus jouer sur les débuts et milieu de saison. On 
# va devoir tester ces hypothèse, en essayant d'abord d'identifier
# un potentiel cycle entre les saisons NBA, où les shoot d'un jour
# donnée dépendent beaucoup du même jour, la saison précédente. 

## Cylce annuel
donnees_journalieres$mois <- as.numeric(format(donnees_journalieres$game_date, "%m"))
#On créé donc une variable mois qu'on ajoute aux données, et
#qui donne pour chaque match à quel mois de la saison il correspond.
#View(donnees_journalieres)

modele_cyclique_annuel_10 <- gam(mean_3pa ~ s(Time,k=4)+s(mois,k=10,bs='cc'), data = donnees_journalieres)
modele_cyclique_annuel_4 <- gam(mean_3pa ~ s(Time,k=4) + s(mois,k=4, bs='cc'), data = donnees_journalieres)

plot(modele_cyclique_annuel_10)
plot(modele_cyclique_annuel_4)

summary(modele_cyclique_annuel_10)
summary(modele_cyclique_annuel_4)

acf(modele_cyclique_annuel_10$residuals)
acf(modele_cyclique_annuel_4$residuals)

pacf(modele_cyclique_annuel_10$residuals)
pacf(modele_cyclique_annuel_4$residuals)

#Interpretation: d'abord, le modèle cyclique avec k = 4. C'est un
#échec : la fonction plot montre que le cycle mensuel qu'on tente
#d'identifier ici est écrasé par le reste du modèle (très proche
# de 0). Pour k = 10, GAM choisit 7.89 de flexibilité. On voit
#apparaître une courbe dont l'amplitude varie entre -2 tirs (creux
# en Juillet) et +4 en Aout. Sauf que si on regarde de plus près,
# les mois de juin à septembre inclus ne font pas partis de la
#saison NBA (pas de matchs). Il s'agit donc seulement de la 
#flexibilité du modèle et de la condition bs = 'cc', qui force 
#les splines à boucler la dernière donnée avec la première, qui 
#laissent apparaître cette absurdité. Les PACF et ACF des résidus
# de ce nouveau modèle, très proches du précédent (tendance seulement)
#montrent bien que ce cycle annuel n'est pas significatif dans la
#dépendance temporelle de nos données. 


## Dépendance de Y_t à Y_(t-1) (auto-regressif d'ordre 1)

donnees_journalieres <- mutate(donnees_journalieres, mean_3pa_lag1 = lag(mean_3pa, n = 1))
#ajout du lag d'ordre 1 aux données
modele_autoreg_1 <- gam(mean_3pa ~ s(Time,k=4) + mean_3pa_lag1, data = donnees_journalieres)

acf(modele_autoreg_1$residuals)
pacf(modele_autoreg_1$residuals)
#On observe une baisse net de la hauteur du plateau qu'on observait
#déjà sur l'ACF de modele_final. De plus, la PACF paraît converger
#vite vers 0 : on est en bonne voie. Cependant, la persistance de
#ce plateau indique que l'auto-regressif d'ordre 1 ne suffit pas
# à décrire les données, bien qu'il fasse partie de la dépendance
#temporelle. On va se tourner vers l'analyse d'un potentiel cycle
#hebdomadaire en NBA, qui serait dû à une programmation favorisant
# les matchs important les week-ends. 


## Cycle hebdomadaire
donnees_journalieres <- donnees_journalieres %>% mutate(jours_semaine = as.factor(weekdays(game_date)))
#on ajoute une colonne pour les jours de la semaine à nos données
#que l'on convertit en facteurs mathématiques avec as.factor()


modele_cyclique_hebdo <- gam(mean_3pa~s(Time, k=4)+jours_semaine,data = donnees_journalieres)

acf(modele_cyclique_hebdo$residuals)
pacf(modele_cyclique_hebdo$residuals)
#Nop, non probant

#Voici une cause probable du "problème" : le lag de 1 jour (auto-
#regressif d'ordre 1) est le modèle explicant le mieux la
# dépendance temporelles des données jursqu'ici. Le problème, c'est
#qu'un lag de 1 fait que, dans le cas particulier du premier jour
#d'une saison s, on regarde le dernier jour de la saison s-1. Or,
#ces deux jours n'ont aucune raison d'être corrélés...


## Supression du saut de saison dans le lag d'ordre 1
# Pour palier à ça, on va empêcher GAM de considérer le "jour
#précédent" lorsqu'il regarde le premier jour d'une saison.

donnees_journalieres <- donnees_journalieres %>% group_by(season_id) %>% mutate(mean_3pa_lag1 = lag(mean_3pa, n = 1)) %>% ungroup()
#maintenant, on retente le GAM: 
modele_autoreg_1 <- gam(mean_3pa ~ s(Time,k=4) + mean_3pa_lag1, data = donnees_journalieres)

acf(modele_autoreg_1$residuals)
pacf(modele_autoreg_1$residuals)

#Interprétation : encore non probant. Cette etape est cruciale pour
#assurer la justesse mathématique de notre auto-regression d'ordre
# 1, mais elle ne permet pas d'améliorer nos ACF et PACF. Le
# plateau persiste, il reste bien un phénomène sous-jacent autre
#qui nous empêche de totalement décrire le moidèle simplement avec
#tendance + autoregression d'ordre 1. 
# Une hypothèse est que les résidus restent temporellement 
#très corrélés d'un jour à l'autre, non pas parce qu'une dépendance 
#générale des 3pts tentés sur plusieurs jours consécutifs existe en NBA,
#mais à cause de la bosse de 1994 à 1997 visible sur nos données.
#Nous avons fait le choix d'une tendance qui n'épouse pas cette
#bosse, car elle est exogène. Elle est due à un changement de règle 
#en NBA qui a ensuite été retirée. Mais prendre une tendance 
#plus proche de cette bosse cause un sur-apprentissage, comme le 
#montre notre validation croisée, qui a bien donné un k = 4 
#(2.999 de flexibilité) pour la modélisation de la tendance. 


#Voici ce qu'on va faire : il faut "expliquer" à notre modèle que
#saisons concernées sont particulières. Pour ça on va commencer
#par les identifier avec une variable bouléenne que l'on va ajouter
#aux données, valant true si la match a eu lieu lors de l'une de
#ces saisons. 


donnees_journalieres <- donnees_journalieres %>% mutate(bosse = season_id %in% c("21994","21995","21996","41994","41995","41996"))


modele_trend_bosse_autoreg_1 <- gam(mean_3pa ~ s(Time,k=4) + mean_3pa_lag1 + bosse, data = donnees_journalieres)

acf(modele_trend_bosse_autoreg_1$residuals)
pacf(modele_trend_bosse_autoreg_1$residuals)

#On a encore ce plateau sur l'ACF qui implique que la prise en
#compte de cette bosse, la tendance et la dépendance au jour t-1
#ne suffisent pas à décrire nos données. Le modèle recherché est
#plus complexe. Ici, il est inutile 
#de sommer des nouvelles equations dans GAM comme nous le faisons
# depuis le début avec les cycle, lag1, lag2 etc... On va passer à
#un modèle AR, voire ARMA si on ne peut déterminer l'ordre p de AR



##  ARMA
#On se reconcentre uniquement sur ce qui le modèle principal : 
#tendance + bosse


modele_trend_bosse <- gam(mean_3pa~s(Time,k=4)+bosse,data=donnees_journalieres)

#on trace la PACF pour obtenir le p du modèle AR (le nombre de lag
#de l'auto-regressif)

pacf(modele_trend_bosse$residuals)
#La pacf met trop longtemps à converger vers 0 pour considérer
#ordre p pour le AR. On va passer à un peu plus complexe : un 
# ARMA(p,q). Pour ça, il nous faut trouver p et q. On va calculer
# différents modèle ARMA pour différentes valeurs de p et q <= 5
# et choisir celui qui minimise le AIC et le BIC, des coefficients,
#favorisant la précision d'un modèle en pénalisant les trop grandes
#valeurs de p et de q. 


pmax= 5
qmax = 5

ordre = expand.grid(p=c(0:pmax),q=c(0:qmax))
ordre <- cbind(ordre[,1],0,ordre[,2])
#View(ordre)

?arima
  
modeles_arima <- apply(ordre,1,arima,x = modele_trend_bosse$residuals,method = c("ML"), SSinit = c("Rossignol2011"), optim.method = "BFGS", include.mean = F)
#apply va appliquer la fonction arima à ordre, ligne par ligne, 
# sur les residus de notre modele, selon les methodes choisies (
#j'ai trouvé ces méthodes d'optimisation dans le corrigé du TP6)


modeles_arima[[10]]$aic

-2*modeles_arima[[10]]$loglik+2*(length(modeles_arima[[10]]$coef)+1)

aic_vecteur<-lapply(modeles_arima,function(x) x$aic)%>%unlist
#calcul de l'AIC pour chaque modele ARMA(p,q)

ordre.opt_aic<-ordre[which.min(aic_vecteur),]
#on obtient p = 4 et q = 3

#Le modèle arima choisi est donc : 
modele_arima_aic = modeles_arima[[which.min(aic_vecteur)]]

acf(modele_arima_aic$residuals)
pacf(modele_arima_aic$residuals)
#Gagné ! Visuellement, toutes les corrélations temporelles de nos
#données ont été absorbés. L'ACF n'est qu'une barre verticale en 0
# et le plateau qui subsistait sur les ordres suivants a disparu.
#Pour la PACF, on dirait bien celle d'un bruit blanc ! 
#Il faut maintenant démontrer proprement (et non visuellement)
#qu'il ne subsite pas de dépendance à un autre phénomène. 

#######test de box pierce
pvalue_BP<-function(model,K)
{
  rho<-acf(model$residuals,lag.max=K,plot=F)$acf[-1]
  n<-model$nobs
  pval<-(1-pchisq(n*sum(rho^2),df=K-length(model$coef)))
  return(pval)
}

pvalue_BP(modele_arima_aic,K=10)

#Malheureusement, le modèle n'est statistiquement pas indépendant
#à 5% près, car la pvalue est < 5%. On va voir si le critère BIC
#donne un modèle ARIMA qui lui valide le test, au cas où. 


bic_vecteur<-lapply(modeles_arima,function(x) -2*x$loglik+log(x$nobs)*length(x$coef))%>%unlist

ordre.opt_bic<-ordre[which.min(bic_vecteur),]
modele_arima_bic = modeles_arima[[which.min(bic_vecteur)]]


acf(modele_arima_bic$residuals)
pacf(modele_arima_bic$residuals)

pvalue_BP(modele_arima_bic, K=10)

#Non plus, le modele minimisant le BIC ne passe pas non plus le 
#test de box pierce

#une idée peut alors être de prendre les autres modèles du vecteur
#modeles_arima, comme le 2è ou le 3è en terme d'AIC par ex, et de
# leur refaire passer le test de box pierce. 

#on essaie avec les deuxièmes en terme d'AIC et de BIC: 
modele_arima_aic_2 = modeles_arima[[order(aic_vecteur)[2]]]
modele_arima_bic_2 = modeles_arima[[order(bic_vecteur)[2]]]
pvalue_BP(modele_arima_aic_2,K=10)
pvalue_BP(modele_arima_bic_2,K=10)


#avec les troisièmes : 
modele_arima_aic_3 = modeles_arima[[order(aic_vecteur)[3]]]
modele_arima_bic_3 = modeles_arima[[order(bic_vecteur)[3]]]
pvalue_BP(modele_arima_aic_3,K=10)
pvalue_BP(modele_arima_bic_3,K=10)


#avec le quatrième : 
modele_arima_aic_4 = modeles_arima[[order(aic_vecteur)[4]]]
modele_arima_bic_4 = modeles_arima[[order(bic_vecteur)[4]]]
pvalue_BP(modele_arima_aic_4,K=10)
pvalue_BP(modele_arima_bic_4,K=10)



#Même les 4 meilleurs au sens de l'AIC ou du BIC ne passent pas
#le test de box pierce. On va boucler sur le tableau modeles_arima
#dans l'ordre des AIC et BIC croissant, pour trouver un modele
#qui passe le test

o = -1
critere <- "aic"

for (i in 5:length(order(aic_vecteur))) {
  o_aic <- order(aic_vecteur)[i]
  o_bic <- order(bic_vecteur)[i]
  
  modele_arima_aic_courant = modeles_arima[[o_aic]]
  if(pvalue_BP(modele_arima_aic_courant,K=10) > 0.05){
    o = o_aic
    break
  }
  modele_arima_bic_courant = modeles_arima[[o_bic]]
  if (pvalue_BP(modele_arima_bic_courant,K=10) > 0.05){
    o = o_bic
    critere <- "bic"
    break
  }
}

o
critere


#Apparement, aucun modele ARMA ne passe le test pour pmax = qmax
# = 5. On va d'abord essayer d'augmenter pmax ou qmax, ce qui va
#profondément augmenter le temps de calcul. Puis on essaiera SARIMA


pmax= 10
qmax = 5

ordre = expand.grid(p=c(0:pmax),q=c(0:qmax))
ordre <- cbind(ordre[,1],0,ordre[,2])


modeles_arima <- apply(ordre,1,arima,x = modele_trend_bosse$residuals,method = c("ML"), SSinit = c("Rossignol2011"), optim.method = "BFGS", include.mean = F)


modeles_arima[[10]]$aic

-2*modeles_arima[[10]]$loglik+2*(length(modeles_arima[[10]]$coef)+1)

aic_vecteur<-lapply(modeles_arima,function(x) x$aic) %>% unlist

ordre.opt_aic<-ordre[which.min(aic_vecteur),]

modele_arima_aic = modeles_arima[[which.min(aic_vecteur)]]

acf(modele_arima_aic$residuals)
pacf(modele_arima_aic$residuals)

#######test de box pierce
pvalue_BP<-function(model,K)
{
  rho<-acf(model$residuals,lag.max=K,plot=F)$acf[-1]
  n<-model$nobs
  pval<-(1-pchisq(n*sum(rho^2),df=K-length(model$coef)))
  return(pval)
}

pvalue_BP(modele_arima_aic,K=10)

modele_arma_nc = modele_arima_aic #nc pour "non-contraint"
#On obtient 0.06, ce qui est au dessus de 0.05. 
#ARMA(8,1) passe donc les tests. En revanche, la complexité
#temporelle pour calculer tous les 66 modèles est énorme. 
#Mais on apprend quelque chose de fondamental : un lag allant
#jusqu'à 8 jours (p=8) a une influence sur la correlation de nos
#données. De plus, le bruit, totalement imprévisible, de la veille
#(q = 1) a une influence aussi sur ma donnée au jour j. Ces 9
# paramètres (lag1,...,lag8 et bruit blanc de la veille) jouent
#un rôle dans ma donnée au temps t. Mais on peut se demander si ils
#jouent TOUS un rôle : sont-ils tous vraiment significatifs ? 

#Pour comprendre à 100% notre modèle actuel : 
# Y_t = Tendance + bosse(94-97) + ARMA(8,0,1) + eps_t
#on va donc se demander si tous les coeffs phi_i (i de 1 à p = 8)
# et theta_1 sont significatif, càd si leur pvalue est < 0.05


library(lmtest)
coeftest(modele_arma_nc)
#le résultat est sans appel : en regardant la colonne Pr(>|z|), on
#constate bien que phi_4,...,phi_7 ne sont pas significatifs, alors
# que phi_1,phi_2 et phi_3, ainsi que phi_8 et theta_1 le sont. 
#On va contraindre ARIMA a prendre ces coefs là à 0, pour obtenir
#un modèle sans coefs à la fois non-significatifs et non-nuls, de
#sorte à éviter un surapprentissage. On commence d'abord par mettre
#phi_5 par exemple, à 0.


contraintes = c(NA, NA, NA, NA, NA, NA, 0, NA, NA) #on force phi_7 
#à valoir 0 d'abord, car c'est celui avec la pire pvalue
initiales = c(0.9985677,0.0938851,-0.0520548,-0.0193094,-0.0210623,0.0288675,-0.0057889,0,-0.9772211)
#on donne pour vect initial, les coeff du modèle non contraint pour
#faciliter la convergence

?arima
modele_arma_contraint = arima(x = modele_trend_bosse$residuals,order = c(8,0,1), fixed = contraintes,method = "CSS",init = initiales, SSinit = c("Rossignol2011"), optim.method = "BFGS", include.mean = F )

#On a désormais un modèle contraint. Vérifions qu'il passe toujours
#le test de box pierce, puis que les coeffs gardés sont 
#significatifs.

pvalue_BP(modele_arma_contraint,K=10)
#C'est gagné : ce modèle contraint (phi_7 nul) passe le test
coeftest(modele_arma_contraint)


#on retente en forçant ce nouveau modèle à partir du précédent,
# ET à avoir un phi_4 nul (le nouveau pire en pvalue)
contraintes <- c(NA, NA, NA, 0, NA, NA, 0, NA, NA)
initiales <- modele_arma_contraint$coef
initiales[4] <- 0


modele_arma_contraintx2 = arima(x = modele_trend_bosse$residuals,order = c(8,0,1), fixed = contraintes,method = "CSS",init = initiales, SSinit = c("Rossignol2011"), optim.method = "BFGS", include.mean = F )

pvalue_BP(modele_arma_contraintx2,K=10)

#Remarque : en essayant de forcer d'autres coeffs, annoncés comme 
#non-significatifs, à 0, on se heurte à des modèles qui ne passent
#plus le test de box pierce. On en déduit que phi_7 est le seul
#ceoff véritablement non-significatif. Les autres sont peut-être
#individuellement peu pertinents, mais assurent collectivement un 
#modèle en lequel on peut avoir confiance. Il ne faut donc pas 
#les contraindre. On conserve donc uniquement celui-ci : 


modele_arma_final <- modele_arma_contraint



## Prédictions : 

#on a maintenant modele_trend_bosse qui contient le GAM donnant la
#tendance et modele_arma_final qui contient le mdoele ARMA 
#décrivant tout ce qui n'est pas du à la tendance : lag,
#saisonnalité... 

#on va créer obtenir des données de matchs qui ne sont pas dans
#dans notre dataset, typiquement les 30 jours suivants, pour 
#tester nos modèles sur de la prédiction


library(hoopR)

logs_nba <- nba_leaguegamelog(season = "2023-24", season_type = "Regular Season")$LeagueGameLog

colnames(logs_nba)

donnees_test <- logs_nba %>%
  mutate(
    GAME_DATE = as.Date(GAME_DATE),
    FG3A = as.numeric(FG3A) 
  ) %>%
  filter(GAME_DATE > as.Date("2023-06-12")) %>%
  group_by(GAME_DATE) %>%
  summarise(mean_3pa = mean(FG3A, na.rm = TRUE)) %>%
  arrange(GAME_DATE)

h <- nrow(donnees_test) #on va prédire la saison 2023-2024

dernier_jour <- max(donnees_journalieres$Time)

donnees_futures <- data.frame(Time = ((dernier_jour+1):(dernier_jour+h)), bosse = FALSE)

?predict
previsions_gam <- predict(modele_trend_bosse, newdata = donnees_futures)
#on a maintenant une prévision déterministe basée sur la tendance


previsions_arma <- as.vector(predict(modele_arma_final,n.ahead = h)$pred)
#et la prévision stochastique basée sur l'ARMA

previsions <- previsions_gam + previsions_arma
#length(previsions)
View(previsions)

#On a donc notre vecteur previsions qui contient, pour chaque jour
#la valeur moyenne des 3 points que notre modèle estime. On va
#évaluer la précision de notre modèle, en regardant la RMSE :


RMSE = sqrt(mean((previsions-donnees_test$mean_3pa)^2))
RMSE
# et la MAE : 

MAE = mean(abs(previsions-donnees_test$mean_3pa))
MAE
#Ainsi, notre modèle prédit la saison suivante de NBA avec une
#erreur de 1.45 3pts tentés en plus ou en moins en moyenne par jour
#du calendrier. Sachant que les matchs de nos jours comptent 
#presque 40 3pts tentés par équipe, une telle MAE confirme que 
#notre modèle est particulièrement robuste en prédiction, même
#sur une periode d'un an. 


## Tracés : 
#Avant de déterminer si cette RMSE est suffisement petite, on va
#tracer quelques courbes. 


tendance_bosse = xts(x = modele_trend_bosse$fitted, order.by = donnees_journalieres$game_date)
partie_gam <- as.numeric(modele_trend_bosse$fitted)
partie_arima <- as.numeric(residuals(modele_trend_bosse)) - as.numeric(residuals(modele_arma_final))
modele_complet <- xts(x = partie_gam + partie_arima, order.by = donnees_journalieres$game_date)
serie_tendance_bosse_arima <- cbind(serie_tendance, tendance_bosse,modele_complet)
serie_modele_complet <- cbind(serie_tempo_3pts,modele_complet)

dygraph(serie_tendance_bosse_arima) %>% dyRangeSelector() %>% dyOptions(colors = c("green","blue","purple","red"))
dygraph(serie_modele_complet) %>% dyRangeSelector() %>% dyOptions(colors = c("green","red"))


## Tracés des prévisions : 
#cette fois on compare nos prévisions aux données réelles, graphiquement


reelles <- xts(x = donnees_test$mean_3pa, order.by = donnees_test$GAME_DATE)
predictions <- xts(x = previsions, order.by = donnees_test$GAME_DATE)
future <- cbind(reelles,predictions)

dygraph(future) %>% dyRangeSelector() %>% dyOptions(colors = c("green", "red"))


library(forecast)
checkresiduals(modele_arma_final)


## Conclusion
#La répartition des résidus paraît effectivement gaussienne, l'ACF
#a quelques barres en dessus du seuil, mais elles sont peu et 
#ne dépassent jamais de beaucoup. La pvalue est de 0.21. On en 
#conclue que le modèle est représentatif des données de façon 
#satisfaisante, et les prévoit avec une erreur moyenne de 1.45
#tirs à 3 points. Etant donnée le nombre de tirs à 3 points 
#tentés par match aujourd'hui (notre précision atteint 96%),
#et les caractéristiques des résidus, proche d'un bruit blanc,
#mise en évidence grace à test_final_residus, on en déduit que 
#l'erreur du modèle est purement aléatoire. Le modèle est validé.
