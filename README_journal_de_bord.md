Approche hybride des deux conventions temporelles. Dans les systèmes de pricing professionnels (et c'est une excellente amélioration que tu pourrais apporter à ton pricer trinomial plus tard), on utilise une approche hybride. On applique le calendrier des jours ouvrés au paramètre de volatilité ($\sigma$), et le calendrier civil aux taux d'intérêt ($r$) et à l'actualisation des dividendes. Cela permet d'expliquer l'effondrement typique de la volatilité implicite observé le vendredi après-midi, les market makers ajustant leurs prix pour neutraliser la perte de temps mort du week-end.

API pour données de bourse : 
		* Yahoo Finance (via la bibliothèque Python yfinance) : Le standard absolu pour le prototypage gratuit. C'est l'outil parfait pour récupérer instantanément 			l'historique OHLC des actions et calculer ta volatilité réalisée. En revanche, c'est inexploitable pour l'historique des options, car l'API ne fournit que les 			cotations du jour en cours, sans aucune profondeur temporelle.

		* ThetaData : Le choix de prédilection pour les développeurs indépendants spécialisés dans les dérivés. Cette API est construite spécifiquement pour les marchés 		d'options. Elle offre un accès très rapide à l'historique complet des Grecques, à la volatilité implicite et aux cotations au niveau de la milliseconde, avec des 		tarifs beaucoup plus accessibles que les acteurs institutionnels classiques.

		* Polygon.io : Le meilleur compromis entre modernité technique et exhaustivité. Leur architecture REST est d'une fluidité exemplaire et la documentation est 			irréprochable. Ils proposent l'historique complet des options américaines, mais il faut prévoir un budget (l'abonnement requis pour l'historique profond n'est pas 		gratuit).

		* Databento : La qualité institutionnelle rendue accessible via un modèle de facturation à la consommation (Pay-as-you-go). Tu ne paies que les gigaoctets exacts 		que tu requiers. C'est l'API idéale si tu souhaites intégrer les véritables bid-ask spreads historiques dans ton modèle pour rendre le calcul de tes coûts de 			friction incontestable.

		* Interactive Brokers (IBKR API) : Le passage logique si ton objectif final est d'automatiser ta stratégie en production. Si tu as un compte financé chez eux, les 		données historiques te coûteront une fraction du prix des fournisseurs externes. Cependant, leur API (TWS) est notoirement archaïque et demande beaucoup de patience 		pour l'extraction de données massives.

Checker WRDS pour données de bourse gratuites

Le terrain de jeu le plus pertinent :  On se concentre sur les ETF actions majeurs (SPY, QQQ, IWM) et les actions individuelles à très forte liquidité (mégacaps). Ces actifs 		proposent des options américaines avec des spreads bid-ask extrêmement serrés, ce qui est crucial pour préserver nos marges sur de faibles volumes.

		Pour opérer efficacement à basse fréquence sur ces sous-jacents, l'approche la plus adaptée consiste à délaisser la volatilité stochastique pure au profit de 			stratégies ciblées :

				* L'arbitrage sur la structure temporelle : Utilisez des spreads calendaires (calendar spreads) ou diagonaux pour exploiter les anomalies de 					tarification de la volatilité implicite entre différentes échéances. Vous profitez de la différence de dépréciation (thêta) et d'exposition à la 				volatilité (véga) sans avoir besoin de rééquilibrer activement votre delta au quotidien.

				* L'arbitrage de prime de risque événementielle : Cette méthode consiste à cibler des actions avant une annonce majeure (comme la publication des 				résultats trimestriels ou des décisions de la FDA). La stratégie implique de vendre une volatilité implicite historiquement surévaluée via des 					positions à risque strictement défini (comme les iron condors), afin de capturer l'effondrement de la volatilité (volatility crush) post-événement.

				* Le trading du "Skew" de volatilité : Sur des ETF comme le SPY, le biais de volatilité rend souvent les puts très chers par rapport aux calls. Des 				stratégies de type risk reversal ou des spreads verticaux asymétriques permettent d'exploiter cette asymétrie de manière quasi-statique.
