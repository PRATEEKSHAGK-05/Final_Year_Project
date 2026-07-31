import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class JayaFeatureSelection:

    def __init__(self,
                 population_size=10,
                 iterations=20,
                 random_state=42):

        self.population_size = population_size
        self.iterations = iterations
        self.random_state = random_state

        np.random.seed(random_state)

    # ---------------------------------------------------
    # Initialize Population
    # ---------------------------------------------------

    def initialize_population(self, n_features):

        population = np.random.randint(
            0,
            2,
            (self.population_size, n_features)
        )

        # Every solution must contain at least one feature

        for i in range(self.population_size):

            if np.sum(population[i]) == 0:

                random_index = np.random.randint(n_features)

                population[i][random_index] = 1

        return population

    # ---------------------------------------------------
    # Fitness Function
    # ---------------------------------------------------

    def fitness(self,
                solution,
                X_train,
                y_train):

        selected_features = np.where(solution == 1)[0]

        if len(selected_features) == 0:

            return 0

        X = X_train[:, selected_features]

        num_classes = len(np.unique(y_train))

        model = XGBClassifier(

            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,

            objective="multi:softmax",

            num_class = num_classes,

            eval_metric="mlogloss",

            random_state=self.random_state,

            

        )
        X_fit, X_val, y_fit, y_val = train_test_split(
            X,
            y_train,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_train
        )

        model.fit(X, y_train)

        prediction = model.predict(X)

        f1 = f1_score(
            y_train,
            prediction,
            average="weighted"
        )

        feature_ratio = len(selected_features) / X_train.shape[1]

        fitness = (0.9 * f1) - (0.1 * feature_ratio)

        return fitness

    # ---------------------------------------------------
    # Select Best Solution
    # ---------------------------------------------------

    def get_best_solution(self,
                          population,
                          X_train,
                          y_train):

        fitness_scores = []

        for solution in population:

            score = self.fitness(
                solution,
                X_train,
                y_train
            )

            fitness_scores.append(score)

        fitness_scores = np.array(fitness_scores)

        best_index = np.argmax(fitness_scores)

        return population[best_index], fitness_scores[best_index]

    # ---------------------------------------------------
    # Select Worst Solution
    # ---------------------------------------------------

    def get_worst_solution(self,
                           population,
                           X_train,
                           y_train):

        fitness_scores = []

        for solution in population:

            score = self.fitness(
                solution,
                X_train,
                y_train
            )

            fitness_scores.append(score)

        fitness_scores = np.array(fitness_scores)

        worst_index = np.argmin(fitness_scores)

        return population[worst_index], fitness_scores[worst_index]

    # ---------------------------------------------------
    # Jaya Update Rule
    # ---------------------------------------------------

    def update_solution(self,
                        current_solution,
                        best_solution,
                        worst_solution):

        r1 = np.random.rand(len(current_solution))

        r2 = np.random.rand(len(current_solution))

        new_solution = (

            current_solution

            + r1 * (best_solution - current_solution)

            - r2 * (worst_solution - current_solution)

        )

        new_solution = np.where(
            new_solution >= 0.5,
            1,
            0
        )

        if np.sum(new_solution) == 0:

            random_index = np.random.randint(len(new_solution))

            new_solution[random_index] = 1

        return new_solution

    # ---------------------------------------------------
    # Start Optimization
    # ---------------------------------------------------

    def fit(self,
            X_train,
            y_train,
            feature_names):

        print("=" * 60)
        print("JAYA FEATURE SELECTION STARTED")
        print("=" * 60)

        number_of_features = X_train.shape[1]

        population = self.initialize_population(
            number_of_features
        )

        best_solution = None

        best_fitness = -999999

                # ---------------------------------------------------
        # Main Jaya Optimization Loop
        # ---------------------------------------------------

        for iteration in range(self.iterations):

            print(f"\nIteration {iteration + 1}/{self.iterations}")

            best_solution, best_score = self.get_best_solution(
                population,
                X_train,
                y_train
            )

            worst_solution, worst_score = self.get_worst_solution(
                population,
                X_train,
                y_train
            )

            if best_score > best_fitness:

                best_fitness = best_score

            print(f"Best Fitness  : {best_score:.4f}")
            print(f"Worst Fitness : {worst_score:.4f}")

            # ---------------------------------------------
            # Update every solution
            # ---------------------------------------------

            for i in range(self.population_size):

                current_solution = population[i]

                new_solution = self.update_solution(
                    current_solution,
                    best_solution,
                    worst_solution
                )

                current_fitness = self.fitness(
                    current_solution,
                    X_train,
                    y_train
                )

                new_fitness = self.fitness(
                    new_solution,
                    X_train,
                    y_train
                )

                if new_fitness > current_fitness:

                    population[i] = new_solution

        print("\nOptimization Completed.")

        # ---------------------------------------------------
        # Final Best Solution
        # ---------------------------------------------------

        best_solution, best_score = self.get_best_solution(
            population,
            X_train,
            y_train
        )

        selected_indices = np.where(best_solution == 1)[0]

        selected_feature_names = feature_names[selected_indices]

        print("\nSelected Features")

        print("-" * 50)

        for feature in selected_feature_names:

            print(feature)

        print("-" * 50)

        print("Total Selected Features :", len(selected_feature_names))

        print("Best Fitness :", round(best_score, 4))

        # ---------------------------------------------------
        # Save Selected Features
        # ---------------------------------------------------

        selected_df = pd.DataFrame({

            "Feature": selected_feature_names

        })

        selected_df.to_csv(

            "results/selected_features.csv",

            index=False

        )

        print("\nselected_features.csv Saved Successfully!")

        return (

            selected_indices,

            selected_feature_names,

            best_score

        )