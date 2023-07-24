#pragma once
//source and updates see on GitHub:  https://github.com/MikhailBelkin/cpp-neuronet
//Author: Mikhail Belkin
//Linked inn: https://www.linkedin.com/in/mike-belkin-1b444950/



#include <vector>
#include <cassert>

#include <iostream>


namespace nuronet {

    class Neuron {
    public:

        using ActivationFunct = double(*)(double);



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

        /// <summary>
        /// GetBulState returned input state if set
        /// </summary>
        /// <returns>bool inputstate</returns>
        bool GetBoolState() {
            return inputstate_ == 1;
        }

        /// <summary>
        /// Constructor taken pointer on ActivationFunction. If sumple F=1 if X>0.5
        /// </summary>
        /// <param name="f">Activation Function</param>
        Neuron(ActivationFunct f) :f_(f) {}


        /// <summary>
        /// AddWeight - set weight of neurons link
        /// </summary>
        /// <param name="value">double value of weight</param>
        void AddWeight(double value) {
            weights_.push_back(value);
        }


        /// <summary>
        /// Activate neuron - activate work of net of nerons from child to parents. Started from index 0 - decision maked main neuron from last layer
        /// All links activate recursive
        /// </summary>
        /// <param name="index"></param>
        /// <returns>return double x = ActivationFunc*weight</returns>
        double Activate(int index) {
            double result = 0;

            //std::string tab(index * 8, ' ');
            //std::cout <<tab << "Activate " << GetId() << "parents =" << index << std::endl;

            for (auto n : links_) {
                // std::cout << tab << "Trying to activate next level:"<<std::endl;
                result += n->Activate(GetId());
                // std::cout << tab << " result = "<<result << std::endl;
            }

            result += inputstate_;//if layer1 inputs is from inputsstate oterwise inputate =0 , and will add nothing

            // std::cout << tab << "Input State is: " << result << ",  wheight=" << weights_[index] << " Activation function return=" << f_(result) * weights_[index] << std::endl;;
            return f_(result) * weights_[index];


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
            std::vector<bool> bool_signals;
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
        bool GetDecision(InputSignals& inp) {

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
