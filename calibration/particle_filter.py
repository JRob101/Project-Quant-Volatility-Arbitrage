"""Particle filter utilities for latent volatility estimation."""

import numpy as np
from scipy.stats import norm

class HestonParticleFilter:
    def __init__(self, num_particles, mu, kappa, theta, xi, rho, dt, initial_var):
        """
        Filtre à particules (SIR) pour estimer la variance cachée du modèle de Heston.
        """
        self.num_particles = num_particles
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt
        
        # Initialisation des particules (scénarios de variance) et de leurs poids uniformes
        self.particles = np.ones(num_particles) * initial_var
        self.weights = np.ones(num_particles) / num_particles

    def step(self, return_t):
        """
        Avance d'un pas de temps en ingérant le rendement log-normal du marché observé.
        """
        # 1. Évaluation de la Vraisemblance (Likelihood)
        # Sous Heston, le rendement attendu est (mu - 0.5*v)*dt et la variance est v*dt
        expected_return = (self.mu - 0.5 * self.particles) * self.dt
        volatility = np.sqrt(np.maximum(self.particles, 1e-6) * self.dt)
        
        # Probabilité d'observer ce rendement de marché sachant la variance de chaque particule
        likelihood = norm.pdf(return_t, loc=expected_return, scale=volatility)
        
        # Mise à jour de l'importance des particules
        self.weights *= likelihood
        self.weights += 1e-12  # Stabilité numérique pour éviter la division par zéro
        self.weights /= np.sum(self.weights)  # Normalisation pour que la somme fasse 1
        
        # 2. Resampling (Rééchantillonnage)
        # Si le nombre de particules efficaces (Effective Sample Size) est trop faible,
        # on élimine les mauvaises particules et on clone les bonnes.
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.num_particles / 2.0:
            indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
            
        # 3. Prédiction (Évolution stochastique pour le jour suivant)
        # Astuce de Javaheri : On extrait le choc du sous-jacent (Z_S) directement depuis 
        # le marché pour conditionner le choc sur la variance, grâce à la corrélation rho.
        Z_S = (return_t - expected_return) / volatility
        Z_perp = np.random.normal(0, 1, self.num_particles)
        
        # Mouvement brownien de la variance (prend en compte l'effet de levier rho)
        dW_v = (self.rho * Z_S + np.sqrt(1 - self.rho**2) * Z_perp) * np.sqrt(self.dt)
        
        # Dynamique CIR de Heston
        v_prev = self.particles
        self.particles = v_prev + self.kappa * (self.theta - v_prev) * self.dt + self.xi * np.sqrt(v_prev) * dW_v
        
        # Troncature pour éviter les variances négatives
        self.particles = np.maximum(self.particles, 1e-6)
        
    def get_estimated_variance(self):
        """Retourne l'espérance mathématique de la variance filtrée."""
        return np.average(self.particles, weights=self.weights)


def append_filtered_volatility(df, params):
    """
    Fonction intégrateur : Parcourt ton DataFrame historique et 
    ajoute l'estimation du filtre à particules.
    """
    # Calcul des log-rendements réels observés
    returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0).values
    
    # Instanciation du filtre avec tes paramètres
    pf = HestonParticleFilter(
        num_particles=2000,  # 2000 scénarios parallèles
        mu=params.get('mu', 0.04), 
        kappa=params.get('kappa', 4.0), 
        theta=params.get('theta', 0.04), 
        xi=params.get('xi', 0.4), 
        rho=params.get('rho', -0.7), 
        dt=1/252, 
        initial_var=params.get('theta', 0.04)
    )
    
    estimated_var = []
    for r in returns:
        if r != 0:
            pf.step(r)
        estimated_var.append(pf.get_estimated_variance())
        
    df_filtered = df.copy()
    df_filtered['Filtered_Variance'] = estimated_var
    # Annualisation de la variance filtrée en volatilité
    df_filtered['Filtered_Vol'] = np.sqrt(df_filtered['Filtered_Variance']) * np.sqrt(252)
    
    return df_filtered
