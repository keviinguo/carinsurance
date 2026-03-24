import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_age_factor(age):
    """Returns frequency factor based on age."""
    if age < 25:
        return 1.8
    elif age <= 60:
        return 1.0
    else:
        return 1.3


def get_accident_factor(accidents):
    """Returns frequency factor based on accident history."""
    return 1.4 ** accidents


def get_mileage_factor(mileage):
    """Returns frequency factor based on annual mileage."""
    if mileage > 15000:
        return 1.2
    else:
        return 1.0


def get_value_factor(car_value):
    """Returns severity factor based on vehicle value."""
    if car_value < 15000:
        return 0.9
    elif car_value <= 30000:
        return 1.0
    else:
        return 1.4


def calculate_premium(age, accidents, mileage, car_value,
                      base_frequency=0.08,
                      base_severity=4000,
                      loading=0.25):
    """
    Calculates expected loss and premium for one driver profile.
    """
    age_factor = get_age_factor(age)
    accident_factor = get_accident_factor(accidents)
    mileage_factor = get_mileage_factor(mileage)
    value_factor = get_value_factor(car_value)

    adjusted_frequency = base_frequency * age_factor * accident_factor * mileage_factor
    adjusted_severity = base_severity * value_factor

    expected_loss = adjusted_frequency * adjusted_severity
    premium = expected_loss * (1 + loading)

    return {
        "Age": age,
        "Accidents": accidents,
        "Mileage": mileage,
        "Car Value": car_value,
        "Adj Frequency": adjusted_frequency,
        "Adj Severity": adjusted_severity,
        "Expected Loss": expected_loss,
        "Premium": premium
    }


# Example driver profiles
drivers = [
    {"age": 22, "accidents": 1, "mileage": 18000, "car_value": 35000},
    {"age": 35, "accidents": 0, "mileage": 12000, "car_value": 22000},
    {"age": 67, "accidents": 2, "mileage": 10000, "car_value": 18000},
    {"age": 28, "accidents": 0, "mileage": 20000, "car_value": 12000},
    {"age": 45, "accidents": 3, "mileage": 16000, "car_value": 40000}
]

results = []
for d in drivers:
    results.append(calculate_premium(d["age"], d["accidents"], d["mileage"], d["car_value"]))

df = pd.DataFrame(results)

print("Insurance Pricing Results")
print(df.round(2))


# Graph 1: Premium by driver
plt.figure(figsize=(8, 5))
plt.bar(range(len(df)), df["Premium"])
plt.xticks(range(len(df)), [f"Driver {i+1}" for i in range(len(df))])
plt.ylabel("Premium ($)")
plt.title("Premium by Driver Profile")
plt.tight_layout()
plt.show()


# Graph 2: Premium vs number of accidents
accident_levels = [0, 1, 2, 3]
premium_values = []

for a in accident_levels:
    result = calculate_premium(age=30, accidents=a, mileage=12000, car_value=25000)
    premium_values.append(result["Premium"])

plt.figure(figsize=(8, 5))
plt.plot(accident_levels, premium_values, marker="o")
plt.xlabel("Number of Accidents")
plt.ylabel("Premium ($)")
plt.title("Premium vs Accident History")
plt.tight_layout()
plt.show()


# Optional Monte Carlo simulation for one driver profile
def simulate_losses(age, accidents, mileage, car_value,
                    n_sims=10000,
                    base_frequency=0.08,
                    base_severity=4000):
    """
    Simulates yearly losses using:
    - Poisson for number of claims
    - Lognormal for claim severity
    """
    age_factor = get_age_factor(age)
    accident_factor = get_accident_factor(accidents)
    mileage_factor = get_mileage_factor(mileage)
    value_factor = get_value_factor(car_value)

    lam = base_frequency * age_factor * accident_factor * mileage_factor
    mean_severity = base_severity * value_factor

    sigma = 0.6
    mu = np.log(mean_severity) - 0.5 * sigma**2

    total_losses = []

    for _ in range(n_sims):
        num_claims = np.random.poisson(lam)

        if num_claims == 0:
            total_losses.append(0)
        else:
            claim_amounts = np.random.lognormal(mean=mu, sigma=sigma, size=num_claims)
            total_losses.append(np.sum(claim_amounts))

    return np.array(total_losses)


sim_losses = simulate_losses(age=22, accidents=1, mileage=18000, car_value=35000)

mean_loss = np.mean(sim_losses)
var_95 = np.percentile(sim_losses, 95)
tvar_95 = sim_losses[sim_losses >= var_95].mean()

print("\nMonte Carlo Simulation Results for Driver 1")
print(f"Mean Simulated Loss: ${mean_loss:.2f}")
print(f"95% VaR: ${var_95:.2f}")
print(f"95% TVaR: ${tvar_95:.2f}")


plt.figure(figsize=(8, 5))
plt.hist(sim_losses, bins=40)
plt.xlabel("Simulated Annual Loss ($)")
plt.ylabel("Frequency")
plt.title("Distribution of Simulated Annual Losses")
plt.tight_layout()
plt.show()