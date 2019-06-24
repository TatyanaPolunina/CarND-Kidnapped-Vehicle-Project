/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#include <random> // Need this for sampling from distributions

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;
  
  std::default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(x, std[0]);

  std::normal_distribution<double> dist_y(y, std[1]);

  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  particles.reserve(num_particles);
  for (int i = 0; i < num_particles; ++ i)
  {
    particles.push_back(Particle(i, dist_x(gen), dist_y(gen), dist_theta(gen), 1));
   // std::cout << "Particle i " << particles[i].x << ' ' << particles[i].y << std::endl;
  }
  
  weights.assign(num_particles, 1);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
    std::default_random_engine gen;
  	double new_theta;
  	double x;
  	double y;
    for (auto& particle: particles)
    {
         double vdt = velocity / yaw_rate;
         new_theta = particle.theta + yaw_rate * delta_t;
         x = particle.x + vdt * (std::sin(new_theta) - std::sin(particle.theta));
      	 y =  particle.y + vdt * (std::cos(particle.theta) - std::cos(new_theta));
         particle.x = std::normal_distribution<double>(x, std_pos[0])(gen);
      	 particle.y = std::normal_distribution<double>(y, std_pos[1])(gen);
         particle.theta = std::normal_distribution<double>(new_theta, std_pos[2])(gen);
         //std::cout << "prediction particle " << particle.x << ' ' << particle.y << ' '  << particle.theta << std::endl; 
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   *  Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
   for (auto& obs: observations)
   {
     auto min_to_pred = [&obs](const LandmarkObs& pred1, const LandmarkObs& pred2) {
       return dist(obs.x, obs.y, pred1.x, pred1.y) < dist(obs.x, obs.y, pred2.x, pred2.y);
     };
     auto iter = std::min_element(predicted.begin(), predicted.end(), min_to_pred);
     obs.id = iter->id;
     
     //std::cout << " min dist " << iter->x << ' ' << iter->y << ' '<<dist(obs.x, obs.y, iter->x, iter->y) << std::endl;
   }
}

LandmarkObs convertToLocalCoordinates(const Particle& particle, float map_x, float map_y)
{
  LandmarkObs prediction;
  double relativeX = map_x - particle.x;
  double relativeY = map_y - particle.y;
  prediction.x = std::cos(particle.theta) * relativeX + std::sin(particle.theta) * relativeY;
  prediction.y = std::cos(particle.theta) * relativeY - std::sin(particle.theta) * relativeX;
  return std::move(prediction);
}

std::pair<double, double> convertToGlobalCoordinates(const Particle& particle, const LandmarkObs& local_landmark)
{
  std::pair<double, double> map_coordinate;
  map_coordinate.first = particle.x + cos(particle.theta) * local_landmark.x - sin(particle.theta) * local_landmark.y;
  map_coordinate.second = particle.y + sin(particle.theta) * local_landmark.x + cos(particle.theta) * local_landmark.y;
  return std::move(map_coordinate);
}

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1.0 / (2 * M_PI * sig_x * sig_y);
  // calculate exponent
  double exponent = pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))
               + pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2));
  //std::cout <<" exp " <<exponent;  
  // calculate weight using normalization terms and exponent
  double weight = gauss_norm * exp(-exponent);
  if (weight == 0)
  {
     std::cout << "error_weight " << weight << ' ' << exponent << ' ' << gauss_norm << std::endl;
     weight =  0.0000001;
  }
  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   *  Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double sum_w = 0.0;
  for (auto& particle: particles)
  {
    std::vector<LandmarkObs> predictions;    
    for (auto& landmark: map_landmarks.landmark_list)
    {
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range)
      {
         LandmarkObs land = convertToLocalCoordinates(particle, landmark.x_f, landmark.y_f);
         land.id = landmark.id_i;
         predictions.push_back(land);
        // std::cout << " pred " << land.x << ' ' << land.y << ' '<<dist(particle.x, particle.y, landmark.x_f, landmark.y_f) << std::endl;
      }
    }
    
    vector<LandmarkObs> part_obs = observations;
    dataAssociation(predictions, part_obs);
    
    particle.weight = 1;
    for (const auto& obs : part_obs)
    {
      const auto global_obs = convertToGlobalCoordinates(particle, obs);
      double weight = multiv_prob(std_landmark[0], std_landmark[1], global_obs.first, global_obs.second, 
                                     map_landmarks.landmark_list[obs.id - 1].x_f, map_landmarks.landmark_list[obs.id - 1].y_f);
      if (weight == 0)
      {
        std::cout << "global obs" << global_obs.first << ' ' << global_obs.second << std::endl;
        std::cout << "particle" << particle.x << ' ' << particle.y<< std::endl;
        std::cout << "obs " << obs.id << ' ' << obs.x << ' ' << obs.y << std::endl;
        if (obs.id == 0  || obs.id < map_landmarks.landmark_list.size())
        {
          std::cout << "ERROR";
        }
        std::cout << "land" <<  map_landmarks.landmark_list[obs.id - 1].x_f << ' ' << map_landmarks.landmark_list[obs.id - 1].y_f << std::endl;
      }
      particle.weight *= weight;
   //   std::cout<< " obs " << obs.x << ' ' << obs.y << std::endl;
   //   std::cout<< "gobs " << global_obs.first << ' ' <<  global_obs.second << std::endl;
   //   std::cout<< "land " <<  map_landmarks.landmark_list[obs.id].x_f << ' ' <<  map_landmarks.landmark_list[obs.id].y_f <<  std::endl;
    }
    sum_w += particle.weight;
  }
  int i = 0;
  std::cout << "Sum " << sum_w << std::endl;
  for (auto& particle: particles)
  {
    particle.weight /= sum_w;
    weights[i++] = particle.weight;
  }
  std::cout << "weights updated " << particles.size() << ' ' << weights.size() << std::endl;
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::cout << "resample Started";
   std::default_random_engine gen;
   std::discrete_distribution<> d(weights.begin(), weights.end());
   double minW = *std::min_element(weights.begin(), weights.end());
   std::cout << "min W: " << minW << std::endl;
   std::vector<Particle> new_particles;
   new_particles.reserve(num_particles);
   for (int i = 0; i < num_particles; ++i)
   {
     new_particles.push_back(particles[d(gen)]);
   }
  particles.swap(new_particles);
  std::cout << "particles updated" << particles.size() << ' ' << new_particles.size() << std::endl;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

