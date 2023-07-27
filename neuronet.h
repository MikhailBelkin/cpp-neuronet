#pragma once
//source and updates see on GitHub:  https://github.com/MikhailBelkin/cpp-neuronet
//Author: Mikhail Belkin
//Linked inn: https://www.linkedin.com/in/mike-belkin-1b444950/



#include <vector>
#include <cassert>

#include <iostream>
#include <math.h>


namespace nuronet {


    double fRand(double fMin, double fMax)
    {
        double f = (double)rand() / RAND_MAX;
        return fMin + f * (fMax - fMin);
    }


    using ActivationFunct = double(*)(double);


    class Neuron {
    public:


        /// <summary>
        /// Constructor taken pointer on ActivationFunction. If sumple F=1 if X>0.5
        /// </summary>
        /// <param name="f">Activation Function</param>
        Neuron(ActivationFunct f) :f_(f) {}


        /// <summary>
        /// SetInputState - set input signals for 1 layer
        /// </summary>
        /// <param name="state">bool state - state of true-yes, false - no</param>
        void SetInputState(bool state) {
            if (state) {
                inputstate_ = 1;
            }
            else {

                inputstate_ = 0;

            }
        }

        double GetInputState() {
            return inputstate_;
        }

        /// <summary>
        /// GetBulState returned input state if set
        /// </summary>
        /// <returns>bool inputstate</returns>
        bool GetBoolState() {
            return inputstate_ == 1;
        }


        /// <summary>
        /// AddWeight - set weight of neurons link
        /// </summary>
        /// <param name="value">double value of weight</param>
        void AddWeight(double value) {
            weights_.push_back(value);
        }


        /// <summary>
        /// Returns vector of weights of Neuron
        /// </summary>
        /// <returns>return vector<<double> wight all weights</returns>
        const std::vector<double>& GetWeights() {
            return weights_;


        }


        /// <summary>
        /// Neuron weight correction
        /// </summary>
        /// <param name="wheight_index">int index of parent neuron, double differeces</param>
        /// <param name="diff"></param>
        void WeightCorretion(int weight_index, double diff) {
            weights_[weight_index] -= diff;
        }


        /// <summary>
        /// Activate neuron - activate work of net of nerons from child to parents. Started from index 0 - decision maked main neuron from last layer
        /// All links activate recursive
        /// </summary>
        /// <param name="index">index - id of parent neuron</param>
        /// <returns>return double x = ActivationFunc*weight</returns>
        double Activate(int index) {
            double result = 0;


            if (links_.size()) { // not layer one - getting inputstate from previos layer;
                for (auto n : links_) {
                    // std::cout << tab << "Trying to activate next level:"<<std::endl;
                    result += n->Activate(GetId());
                    // std::cout << tab << " result = "<<result << std::endl;
                }
                inputstate_ = f_(result); //save result as inputstate as cash_value;
            }
            
            

           //  std::cout << tab << "Input State is: " << result << ",  wheight=" << weights_[index] << " Activation function return=" << f_(result) * weights_[index] << std::endl;;
            return inputstate_ * weights_[index];


        }


        double Learn(int parent_index,  double weights_delta, double learning_rate) {
            
            

            double error_diff;
            double weights_delta_for_next_layer;
            if (parent_index!=-1) { //Magic number. Not a decision neuron
                
                //std::cout << "weight[" << parent_index << "]=" << weights_[parent_index]<<std::endl;

              //  std::cout<<"Old weight="<< weights_[parent_index]<<" new weight=";
                weights_[parent_index] -= weights_delta*inputstate_*learning_rate;
                
              //  std::cout << weights_[parent_index] << " weight diff=" << weights_delta * inputstate_ * learning_rate << std::endl;
                error_diff = weights_[parent_index] * weights_delta;
              //  std::cout << "weight delta for next layer=" << weights_delta << " error diff=" << error_diff << std::endl;
                weights_delta_for_next_layer = error_diff;

            }
            else {
                weights_delta_for_next_layer = weights_delta;
                parent_index = 0;//return_paren_index to normal from -1;
            }

            if (links_.size()) { //layer0 dont have links;
                inputstate_ = 0;
                

                for (auto n : links_) {

                    inputstate_ += n->Learn(GetId(),  weights_delta_for_next_layer, learning_rate);

                }
                inputstate_ = f_(inputstate_);
            }
            return  inputstate_ * weights_[parent_index];
            

        }


        /// <summary>
        /// Reset all weights in random values
        /// </summary>
        void ResetAllWeights(){
            for (auto w : weights_) {
                w = fRand(0, 10);
            }
            for (auto n : links_) {
                
                n->ResetAllWeights();
                
            }
        
        }

        friend class NeuronsLayer;
        friend class SkyNet;

    private:

        /// <summary>
        /// MakeLinkFrom - using NeuronLayers for linking neurons from previos layer to neurons of current laeyer
        /// </summary>
        /// <param name="from"></param>
        void MakeLinkFrom(Neuron& from) {
            links_.push_back(&from);

        }

        /// <summary>
        /// SetId - set id from layer while adding new neurons. 
        /// </summary>
        /// <param name="id">int - id number</param>
        void SetId(int id) {
            id_ = id;
        }

        /// <summary>
        /// GetID - returns id on current neuron
        /// </summary>
        /// <returns>return id of current neuron</returns>
        int GetId() {
            return id_;
        }




        int id_ = 0; //id of current neuron needed for recursion

        double inputstate_ = 0; //inputstate - needed for 1 layer as input signal


        std::vector<Neuron*> links_;//links to oter neurons


        std::vector<double> weights_;//weight for each link

        ActivationFunct f_;//ActivationFunc - set by user for every neuron
    };



    /// <summary>
    /// Neuron layer - set of neurons. Usualy is input_layer, hidden_layer, output_layer
    /// </summary>
    class NeuronsLayer {
    public:

        /// <summary>
        /// constructor by default
        /// </summary>
        NeuronsLayer() {}

        /// <summary>
        /// AddNeuron - add new neuron into the current layer
        /// </summary>
        /// <param name="neuron">link of neurons. Invalidate after adding</param>
        void AddNeron(Neuron& neuron) {
            neurons_.push_back(std::move(neuron));
            neurons_.back().SetId(neurons_.size() - 1);
        }


        /// <summary>
        /// GetNeuronCount - returns count of neuron in current layer
        /// </summary>
        /// <returns>int - count of neurons in layer</returns>
        int GetNeuronCount() {
            return neurons_.size();
        }

        /// <summary>
        /// GetNeuron - get a neuron. Acessed by index
        /// </summary>
        /// <param name="index">int index - number of neuron in adding order</param>
        /// <returns>link to Neuron</returns>
        Neuron& GetNeuron(int index) {

            ///check index range
            if (index >= 0 && index < neurons_.size()) {
                return neurons_[index];
            }
            else {
                assert("Out of range nuerons");
            }

        }

        /// <summary>
        /// Same the non const
        /// </summary>
        /// <param name="index">int index - number of neuron</param>
        /// <returns>const Neuron& conatstant link to neuron</returns>
        const Neuron& GetNeuron(int index) const {
            if (index > 0 && index < neurons_.size()) {
                return neurons_.at(index);
            }
            else {
                assert("Out of range nuerons");
            }

        }

    private:

        std::vector<Neuron> neurons_;//array of neurons


    };

    /// <summary>
    /// class Skynet is net of nurons
    /// </summary>
    class SkyNet {
    public:

        /// <summary>
        /// input signals format
        /// </summary>
        struct InputSignals {
            InputSignals(std::vector<bool>& v):bool_signals(v) {
                
            }
            
            std::vector<bool>& bool_signals;
        };


        /// <summary>
        /// Default constructor
        /// </summary>
        SkyNet() :decision_(nullptr) {}


        /// <summary>
        /// GetDicision - return decision after activate all neurons of net
        /// </summary>
        /// <param name="inp">SkyNet::InputSignals.signals - vector<bool></param>
        /// <returns></returns>
        /// 
        double GetDecision(InputSignals& inp) {

            if (inp.bool_signals.size() != layers_[0].GetNeuronCount()) {
                //count of input signals must be equal number of neurons in input layer
                assert("Input signals not mathing with layer1");
            }

            // Set input signal for ever nuron of input layer
            for (int i = 0; i < layers_[0].GetNeuronCount(); i++) {
                layers_[0].GetNeuron(i).SetInputState(inp.bool_signals[i]);
            }
            // return result of recursive activate nerons strting from main decision neuron
            return decision_->Activate(0);
        }


        void ResetAllWeights() {
            decision_->ResetAllWeights();
        }


        void PrintDump() {
            std::cout << "********************Print dumping od Network:" << std::endl;
            int layer_num = 0;
            for (auto l : layers_) {
                std::cout << "--------Layer " << layer_num++ << ":" << std::endl;
                for (int i = 0; i < l.GetNeuronCount(); i++) {
                    std::cout << "Neuron[" << i << "]: ";
                    auto weights = l.GetNeuron(i).GetWeights();
                    int weights_count = 0;
                    for (auto w : weights) {
                        std::cout << "Wheight[" << weights_count++ << "]=" << w<<" ; ";
                    }
                    std::cout << "Inputstate=" << l.GetNeuron(i).GetInputState() << std::endl;
                }
            }
            std::cout << "*************end of dump***************" << std::endl;

        }



        void Learning(std::vector<InputSignals>& inp, std::vector<double> expected_decision, int epoche_num, double learning_rate) {

            if (inp.size() != expected_decision.size()) {
                assert("Size of vectors for learning is not equal");

            }




            for (auto x : inp) {
                if (x.bool_signals.size() != layers_[0].GetNeuronCount()) {
                    //count of input signals must be equal number of neurons in input layer
                    assert("Input signals not mathing with layer1");
                }
            }

            ResetAllWeights();


            for (int epoche = 0; epoche < epoche_num; epoche++) {
               for (int i = 0; i < inp.size();i++) {
                    double actual_decision = GetDecision(inp[i]);
                    double error_decision = (actual_decision - expected_decision[i]);
                    double weights_delta = error_decision * (actual_decision * (1 - actual_decision));
                   
                    double result_of_learning = decision_->Learn(-1, weights_delta, learning_rate);
                    
                }
            }

        }

        
        /// <summary>
        /// AddLayer - adds new layer into net
        /// </summary>
        /// <param name="layer">NeuronLayer - new set of neurons</param>
        void AddLayer(NeuronsLayer& layer) {

            layers_.push_back(layer);
            int current_layer = layers_.size() - 1;


            if (layers_.size() > 1) {


                // Make links with all of neurons from previos layer to new layer
                for (int i = 0; i < layers_[current_layer].GetNeuronCount(); i++) {
                    for (int j = 0; j < layers_[current_layer - 1].GetNeuronCount(); j++) {

                        layers_[current_layer].GetNeuron(i).MakeLinkFrom(
                            layers_[current_layer - 1].GetNeuron(j)
                        );

                    }

                }
            }

            //If layer have only one neuron - it is neron for main decision - top of tree
            if (layer.GetNeuronCount() == 1) {
                decision_ = &layers_[current_layer].GetNeuron(0);

            }



        }



    private:

        std::vector<NeuronsLayer> layers_;/// array of layers

        Neuron* decision_; // pointer to main decision neron to start activate process
    };


}// end of namespace
