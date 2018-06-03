# 04/18 Création Morgan SCAO

set @dtref = '2018-04-01';
set @dtmoins1 = @dtref - interval 1 year;
set @dtmoins2 = @dtref - interval 2 year;
set @dtmoins3 = @dtref - interval 3 year;
set @dtmoins4 = @dtref - interval 4 year;

# Nettoyage
delete from tmp.scores_stats;


# Liste des SIREN
insert into tmp.scores_stats (siren, procol, indiScore, indiScoreDate, encours)
(
select distinct siren, procol, indiScore20, indiScoreDate, encours
from jo.scores_surveillance aa
#where substr(CJ,1,1)<>7
order by siren
);
# On met éventuellement à jour avec la table historique
update tmp.scores_stats tt inner join historiques.scores_surveillance aa on aa.siren=tt.siren 
and aa.dateUpdate = (select max(dateUpdate) from historiques.scores_surveillance bb 
	where bb.siren=aa.siren and bb.indiScoreDate<=@dtref)
and (aa.indiScoreDate>tt.indiScoreDate or tt.indiScoreDate>@dtref)
set 
tt.indiScore=aa.indiScore20,
tt.indiScoreDate=aa.indiScoreDate,
tt.procol=aa.procol,
tt.encours=aa.encours;

# Liste des évènements des 6 derniers mois
update tmp.scores_stats tt set lastTypeEven=
(select GROUP_CONCAT(typeeven) from jo.bodacc_detail b1 where tt.siren=b1.siren and dateinsert>=
((select max(dateinsert) from jo.bodacc_detail b2 where b1.siren=b2.siren)-interval 180 day)
);
update tmp.scores_stats tt set lastTypeEven2=
(select GROUP_CONCAT(typeeven) from jo.annonces b1 where tt.siren=b1.siren and dateinsert>=
((select max(dateinsert) from jo.annonces b2 where b1.siren=b2.siren)-interval 180 day)
);


# Historique 1 an
# On commence par la table historique
update tmp.scores_stats tt inner join historiques.scores_surveillance aa on aa.siren=tt.siren 
and aa.dateUpdate = (select max(dateUpdate) from historiques.scores_surveillance bb 
	where bb.siren=aa.siren and bb.indiScoreDate<=@dtmoins1)
set 
tt.indiScoreMoins1=aa.indiScore20,
tt.indiScoreDateMoins1=aa.indiScoreDate,
tt.procolMoins1=aa.procol,
tt.encoursMoins1=aa.encours;
# Attention au dernier calcul (dans jo)
update tmp.scores_stats tt inner join jo.scores_surveillance aa on aa.siren=tt.siren 
and aa.dateUpdate = (select max(dateUpdate) from jo.scores_surveillance bb 
	where bb.siren=aa.siren and bb.indiScoreDate<=@dtmoins1)
and aa.indiScoreDate>tt.indiScoreDateMoins1
set 
tt.indiScoreMoins1=aa.indiScore20,
tt.indiScoreDateMoins1=aa.indiScoreDate,
tt.procolMoins1=aa.procol,
tt.encoursMoins1=aa.encours;

# Historique 2 an
# On commence par la table historique
update tmp.scores_stats tt inner join historiques.scores_surveillance aa on aa.siren=tt.siren 
and aa.dateUpdate = (select max(dateUpdate) from historiques.scores_surveillance bb 
	where bb.siren=aa.siren and bb.indiScoreDate<=@dtmoins2)
set 
tt.indiScoreMoins2=aa.indiScore20,
tt.indiScoreDateMoins2=aa.indiScoreDate,
tt.procolMoins2=aa.procol,
tt.encoursMoins2=aa.encours;
# Attention au dernier calcul (dans jo)
update tmp.scores_stats tt inner join jo.scores_surveillance aa on aa.siren=tt.siren 
and aa.dateUpdate = (select max(dateUpdate) from jo.scores_surveillance bb 
	where bb.siren=aa.siren and bb.indiScoreDate<=@dtmoins2)
and aa.indiScoreDate>tt.indiScoreDateMoins2
set 
tt.indiScoreMoins2=aa.indiScore20,
tt.indiScoreDateMoins2=aa.indiScoreDate,
tt.procolMoins2=aa.procol,
tt.encoursMoins2=aa.encours;

# Historique 3 an
# On commence par la table historique
update tmp.scores_stats tt inner join historiques.scores_surveillance aa on aa.siren=tt.siren 
and aa.dateUpdate = (select max(dateUpdate) from historiques.scores_surveillance bb 
	where bb.siren=aa.siren and bb.indiScoreDate<=@dtmoins3)
set 
tt.indiScoreMoins3=aa.indiScore20,
tt.indiScoreDateMoins3=aa.indiScoreDate,
tt.procolMoins3=aa.procol,
tt.encoursMoins3=aa.encours;
# Attention au dernier calcul (dans jo)
update tmp.scores_stats tt inner join jo.scores_surveillance aa on aa.siren=tt.siren 
and aa.dateUpdate = (select max(dateUpdate) from jo.scores_surveillance bb 
	where bb.siren=aa.siren and bb.indiScoreDate<=@dtmoins3)
and aa.indiScoreDate>tt.indiScoreDateMoins3
set 
tt.indiScoreMoins3=aa.indiScore20,
tt.indiScoreDateMoins3=aa.indiScoreDate,
tt.procolMoins3=aa.procol,
tt.encoursMoins3=aa.encours;

# Historique 4 an
# On commence par la table historique
update tmp.scores_stats tt inner join historiques.scores_surveillance aa on aa.siren=tt.siren 
and aa.dateUpdate = (select max(dateUpdate) from historiques.scores_surveillance bb 
	where bb.siren=aa.siren and bb.indiScoreDate<=@dtmoins4)
set 
tt.indiScoreMoins4=aa.indiScore20,
tt.indiScoreDateMoins4=aa.indiScoreDate,
tt.procolMoins4=aa.procol,
tt.encoursMoins4=aa.encours;
# Attention au dernier calcul (dans jo)
update tmp.scores_stats tt inner join jo.scores_surveillance aa on aa.siren=tt.siren 
and aa.dateUpdate = (select max(dateUpdate) from jo.scores_surveillance bb 
	where bb.siren=aa.siren and bb.indiScoreDate<=@dtmoins4)
and aa.indiScoreDate>tt.indiScoreDateMoins4
set 
tt.indiScoreMoins4=aa.indiScore20,
tt.indiScoreDateMoins4=aa.indiScoreDate,
tt.procolMoins4=aa.procol,
tt.encoursMoins4=aa.encours;


#select * from tmp.scores_stats where substr(ii_CJ,1,1)=5;
