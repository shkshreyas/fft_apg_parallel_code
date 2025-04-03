#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <time.h>

typedef struct {
    int population_size;
    int chromosome_length;
    double mutation_rate;
    double crossover_rate;
    int max_generations;
    int num_threads;
    int adaptive_threading;
    int locality_aware;
    int auto_tune_interval;
} GA_Config;

typedef struct {
    int* genes;
    double fitness;
    int thread_affinity;
} Chromosome;

void initialize_population(Chromosome* population, GA_Config* config);
void evaluate_fitness(Chromosome* population, GA_Config* config);
void selection(Chromosome* population, Chromosome* new_population, GA_Config* config);
void crossover(Chromosome* population, GA_Config* config);
void mutation(Chromosome* population, GA_Config* config);
void auto_tune_parameters(GA_Config* config, int generation, double avg_fitness);
void optimize_thread_distribution(Chromosome* population, GA_Config* config);

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));
    
    GA_Config config = {
        .population_size = 1000,
        .chromosome_length = 100,
        .mutation_rate = 0.01,
        .crossover_rate = 0.8,
        .max_generations = 500,
        .num_threads = omp_get_max_threads(),
        .adaptive_threading = 1,
        .locality_aware = 1,
        .auto_tune_interval = 10
    };
    
    Chromosome* population = (Chromosome*)malloc(config.population_size * sizeof(Chromosome));
    Chromosome* new_population = (Chromosome*)malloc(config.population_size * sizeof(Chromosome));
    
    initialize_population(population, &config);
    
    double start_time = omp_get_wtime();
    double best_fitness = 0.0;
    double avg_fitness = 0.0;
    
    printf("Starting GA with %d threads\n", config.num_threads);
    
    for (int generation = 0; generation < config.max_generations; generation++) {
        evaluate_fitness(population, &config);
        
        avg_fitness = 0.0;
        best_fitness = population[0].fitness;  // Corrected here
        
        #pragma omp parallel for reduction(+:avg_fitness) schedule(dynamic)
        for (int i = 0; i < config.population_size; i++) {
            avg_fitness += population[i].fitness;
            #pragma omp critical
            {
                if (population[i].fitness > best_fitness) {
                    best_fitness = population[i].fitness;
                }
            }
        }
        avg_fitness /= config.population_size;
        
        if (generation % config.auto_tune_interval == 0) {
            auto_tune_parameters(&config, generation, avg_fitness);
        }
        
        if (config.locality_aware) {
            optimize_thread_distribution(population, &config);
        }
        
        selection(population, new_population, &config);
        
        Chromosome* temp = population;
        population = new_population;
        new_population = temp;
        
        crossover(population, &config);
        mutation(population, &config);
        
        printf("Generation %d: Best Fitness = %.4f, Avg Fitness = %.4f\n", generation, best_fitness, avg_fitness);
    }
    
    double elapsed_time = omp_get_wtime() - start_time;
    printf("GA completed in %.2f seconds\n", elapsed_time);
    
    for (int i = 0; i < config.population_size; i++) {
        free(population[i].genes);
        free(new_population[i].genes);
    }
    free(population);
    free(new_population);
    
    return 0;
}

void initialize_population(Chromosome* population, GA_Config* config) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < config->population_size; i++) {
        population[i].genes = (int*)malloc(config->chromosome_length * sizeof(int));
        population[i].thread_affinity = omp_get_thread_num();
        
        for (int j = 0; j < config->chromosome_length; j++) {
            population[i].genes[j] = rand() % 2;
        }
        population[i].fitness = 0.0;
    }
}

void evaluate_fitness(Chromosome* population, GA_Config* config) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < config->population_size; i++) {
        if (config->locality_aware && omp_get_thread_num() != population[i].thread_affinity) {
            if (omp_get_num_threads() > 1) {
                continue;
            }
        }
        
        double sum = 0.0;
        for (int j = 0; j < config->chromosome_length; j++) {
            sum += population[i].genes[j];
        }
        population[i].fitness = sum / config->chromosome_length;
    }
}

void selection(Chromosome* population, Chromosome* new_population, GA_Config* config) {
    const int tournament_size = 5;
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < config->population_size; i++) {
        int best_idx = rand() % config->population_size;
        double best_fit = population[best_idx].fitness;
        
        for (int j = 1; j < tournament_size; j++) {
            int idx = rand() % config->population_size;
            if (population[idx].fitness > best_fit) {
                best_idx = idx;
                best_fit = population[idx].fitness;
            }
        }
        
        new_population[i].genes = (int*)malloc(config->chromosome_length * sizeof(int));
        memcpy(new_population[i].genes, population[best_idx].genes, config->chromosome_length * sizeof(int));
        new_population[i].fitness = population[best_idx].fitness;
        new_population[i].thread_affinity = omp_get_thread_num();
    }
}

void crossover(Chromosome* population, GA_Config* config) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < config->population_size; i += 2) {
        if (i + 1 < config->population_size && (double)rand() / RAND_MAX < config->crossover_rate) {  // Corrected here
            int crossover_point = rand() % config->chromosome_length;
            
            for (int j = crossover_point; j < config->chromosome_length; j++) {
                int temp = population[i].genes[j];
                population[i].genes[j] = population[i + 1].genes[j];
                population[i + 1].genes[j] = temp;
            }
        }
    }
}

void mutation(Chromosome* population, GA_Config* config) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < config->population_size; i++) {
        for (int j = 0; j < config->chromosome_length; j++) {
            if ((double)rand() / RAND_MAX < config->mutation_rate) {
                population[i].genes[j] = 1 - population[i].genes[j];
            }
        }
    }
}

void auto_tune_parameters(GA_Config* config, int generation, double avg_fitness) {
    if (avg_fitness > 0.8) {
        config->mutation_rate = fmin(0.1, config->mutation_rate * 1.5);
    } else if (avg_fitness < 0.3) {
        config->mutation_rate = fmax(0.001, config->mutation_rate * 0.8);
    }
    
    if (config->adaptive_threading) {
        int optimal_threads;
        
        if (generation < config->max_generations / 10) {
            optimal_threads = omp_get_max_threads();
        } else if (generation > config->max_generations * 0.8) {
            optimal_threads = omp_get_max_threads() / 2;
            if(optimal_threads < 1)
                optimal_threads = 1;
        } else {
            optimal_threads = (omp_get_max_threads() * 3) / 4;
            if(optimal_threads < 2)
                optimal_threads = 2;
        }
        
        if (config->num_threads != optimal_threads) {
            config->num_threads = optimal_threads;
            omp_set_num_threads(optimal_threads);
            printf("Auto-tuning: Adjusted thread count to %d\n", optimal_threads);
        }
    }
}

void optimize_thread_distribution(Chromosome* population, GA_Config* config) {
    int* chromosomes_per_thread = (int*)calloc(config->num_threads, sizeof(int));
    
    for (int i = 0; i < config->population_size; i++) {
        if (population[i].thread_affinity < config->num_threads) {
            chromosomes_per_thread[population[i].thread_affinity]++;
        }
    }
    
    int min_thread = 0, max_thread = 0;
    for (int i = 1; i < config->num_threads; i++) {
        if (chromosomes_per_thread[i] < chromosomes_per_thread[min_thread]) {
            min_thread = i;
        }
        if (chromosomes_per_thread[i] > chromosomes_per_thread[max_thread]) {
            max_thread = i;
        }
    }
    
    if (chromosomes_per_thread[max_thread] > chromosomes_per_thread[min_thread] * 1.5) {
        int to_redistribute = (chromosomes_per_thread[max_thread] - chromosomes_per_thread[min_thread]) / 2;
        
        int redistributed = 0;
        for (int i = 0; i < config->population_size && redistributed < to_redistribute; i++) {
            if (population[i].thread_affinity == max_thread) {
                population[i].thread_affinity = min_thread;
                redistributed++;
            }
        }
    }
    
    free(chromosomes_per_thread);
}
