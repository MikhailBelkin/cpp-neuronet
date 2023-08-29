#pragma once
//source and updates see on GitHub:  https://github.com/MikhailBelkin/cpp-neuronet
//Author: Mikhail Belkin
//Linked inn: https://www.linkedin.com/in/mike-belkin-1b444950/



#include <vector>
#include <cassert>

#include <iostream>
#include <math.h>
#include "neural_net.pb.h"

namespace neuronet {


    using ActivationFunct = double(*)(double);


    double fRand(double fMin, double fMax);
    double SigmaFunction(double d);

    template<typename InputT>
    class Neuron {
    public:

        /// <summary>
        /// Default constructor is working with SigmaFunction by default
        /// </summary>
        Neuron():f_(&SigmaFunction) {


        }

        /// <summary>
        /// Constructor taken pointer on ActivationFunction. 
        /// </summary>
        /// <param name="f">Activation Function</param>
        Neuron(ActivationFunct f) :f_(f) {}


        /// <summary>
        /// Constructor wight weights and Activation function. F is SigmaFunction by dafault
        /// </summary>
        /// <param name="num_of_weights"></param>
        Neuron(int num_of_weights);


        /// <summary>
        /// SetInputState - set input signals for 1 layer
        /// </summary>
        /// <param name="state">bool state - state of true-yes, false - no</param>
        void SetInputState(InputT state);

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
        double Activate(int index);


        double Learn(int parent_index, double weights_delta, double learning_rate);


        /// <summary>
        /// Reset all weights in random values
        /// </summary>
        void ResetAllWeights();
        
        template <typename InputT>
        friend class NeuronsLayer;

        template <typename InputT>
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
    template <typename InputT>
    class NeuronsLayer {
    public:

        /// <summary>
        /// constructor by default
        /// </summary>
        NeuronsLayer() {}
        NeuronsLayer(int neurons_num, std::vector<Neuron<InputT>>& n) {
            neurons_.resize(neurons_num);
            std::copy_n(n.begin(), neurons_num, neurons_.begin());
        }

        /// <summary>
        /// AddNeuron - add new neuron into the current layer
        /// </summary>
        /// <param name="neuron">link of neurons. Invalidate after adding</param>
        void AddNeron(Neuron<InputT>& neuron) {
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
        Neuron<InputT>& GetNeuron(int index);

        /// <summary>
        /// GetNeuron - get a neuron. Acessed by index. Const
        /// </summary>
        /// <param name="index">int index - number of neuron</param>
        /// <returns>const Neuron& conatstant link to neuron</returns>
        const Neuron<InputT>& GetNeuron(int index) const;


        const std::vector<Neuron<InputT>> NeuronsLayer<InputT>::GetNeurons() const;

    private:

        std::vector<Neuron<InputT>> neurons_;//array of neurons


    };

    /// <summary>
    /// class Skynet is net of nurons
    /// </summary>
    template <typename T>
    class SkyNet {
    public:

        /// <summary>
        /// input signals format for layer 1
        /// </summary>
        
        struct InputSignals {
            InputSignals(std::vector<T>& v):signals(v) {
                
            }
            
            std::vector<T>& signals;
        };


        /// <summary>
        /// Default constructor
        /// </summary>
        //SkyNet() :decision_(nullptr) {}


        /// <summary>
        /// GetDicision - return decision after activate all neurons of net
        /// </summary>
        /// <param name="inp">SkyNet::InputSignals.signals - vector<bool></param>
        /// <returns></returns>
        std::vector<double> GetDecision(InputSignals& inp);


        /// <summary>
        /// Resets weights of all neurons in the net
        /// </summary>
        void ResetAllWeights();



        /// <summary>
        /// Dubug dump of all neurons to ostream
        /// </summary>
        void PrintDump(std::ostream& out);

        void Serialize(std::ostream& out);

        void DeSerialize(std::istream& in);

        /// <summary>
        /// Learn Neural by backpropagation method
        /// </summary>
        /// <param name="inp">set of input signals for layer1</param>
        /// <param name="expected_decision">set of expected decisions for current input set. Depended from number of decisions nuerons. If you have only one - decision is 0..1, if you have "num" - decision is 0..num</param>
        /// <param name="epoche_num">count of cycles(epoche) for learning.Usualy 1000-3000</param>
        /// <param name="learning_rate">learning rate. Ususaly about 0.1</param>
        void Learning(std::vector<InputSignals>& inp, std::vector<double> expected_decision, int epoche_num, double learning_rate);

        
        /// <summary>
        /// AddLayer - adds new layer into net
        /// </summary>
        /// <param name="layer">NeuronLayer - new set of neurons</param>
        /// <param name="decision_layer">bool flag for last layer - it is decision neurons</param>
        void AddLayer(NeuronsLayer<T>& layer, bool decision_layer = 0);



    private:

        std::vector<NeuronsLayer<T>> layers_;/// array of layers

        std::vector<Neuron<T>*> decision_; // pointer to main decision neron to start activate process
    };





    template<typename InputT>
    Neuron<InputT>::Neuron(int num_of_weights) {
        for (int i = 0; i < num_of_weights; i++) {
            AddWeight(fRand(0, 1));
        }
        f_ = &SigmaFunction;

    }


    template<typename InputT>
    void Neuron<InputT>::SetInputState(InputT state) {
        if (!state) {
            inputstate_ = 0;
            return;
        }
        inputstate_ = 1 / sizeof(state) * state;

    }

    template<typename InputT>
    double Neuron<InputT>::Activate(int index) {
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

    template<typename InputT>
    double Neuron<InputT>::Learn(int parent_index, double weights_delta, double learning_rate) {



        double error_diff;
        double weights_delta_for_next_layer;
        if (parent_index != -1) { //Magic number. Not a decision neuron

            //std::cout << "weight[" << parent_index << "]=" << weights_[parent_index]<<std::endl;

          //  std::cout<<"Old weight="<< weights_[parent_index]<<" new weight=";
            weights_[parent_index] -= weights_delta * inputstate_ * learning_rate;

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

                inputstate_ += n->Learn(GetId(), weights_delta_for_next_layer, learning_rate);

            }
            inputstate_ = f_(inputstate_);
        }
        return  inputstate_ * weights_[parent_index];


    }

    template<typename InputT>
    void Neuron<InputT>::ResetAllWeights() {
        for (auto w : weights_) {
            w = fRand(0, 10);
        }
        for (auto n : links_) {

            n->ResetAllWeights();

        }

    }


    template <typename InputT>
    Neuron<InputT>& NeuronsLayer<InputT>::GetNeuron(int index) {

        ///check index range
        if (index >= 0 && index < neurons_.size()) {
            return neurons_[index];
        }
        else {
            assert("Out of range nuerons");
        }

    }


    template <typename InputT>
    const Neuron<InputT>& NeuronsLayer<InputT>::GetNeuron(int index) const {
        if (index > 0 && index < neurons_.size()) {
            return neurons_.at(index);
        }
        else {
            assert("Out of range nuerons");
        }

    }


    template <typename InputT>
    const std::vector<Neuron<InputT>> NeuronsLayer<InputT>::GetNeurons() const {
        return neurons_;

    }



    template <typename T>
    std::vector<double> SkyNet<T>::GetDecision(InputSignals& inp) {
        std::vector<double> result;
        if (inp.signals.size() != layers_[0].GetNeuronCount()) {
            //count of input signals must be equal number of neurons in input layer
            assert("Input signals not mathing with layer1");
        }

        // Set input signal for ever nuron of input layer
        for (int i = 0; i < layers_[0].GetNeuronCount(); i++) {
            layers_[0].GetNeuron(i).SetInputState(inp.signals[i]);
        }
        // return result of recursive activate nerons strting from main decision neuron
        for (int i = 0; i < layers_[layers_.size() - 1].GetNeuronCount(); i++) {
            result.push_back(decision_[i]->Activate(0));
        }
        return result;

    }



    template <typename T>
    void SkyNet<T>::ResetAllWeights() {
        for (int i = 0; i < layers_[layers_.size() - 1].GetNeuronCount(); i++) {
            decision_[i]->ResetAllWeights();
        }
    }


    template <typename T>
    void SkyNet<T>::PrintDump(std::ostream& out) {
        out << "********************Print dumping of Network:" << std::endl;
        int layer_num = 0;
        for (auto l : layers_) {
            out << "--------Layer " << layer_num++ << ":" << std::endl;
            for (int i = 0; i < l.GetNeuronCount(); i++) {
                out << "Neuron[" << i << "]: ";
                auto weights = l.GetNeuron(i).GetWeights();
                int weights_count = 0;
                for (auto w : weights) {
                    out << "Wheight[" << weights_count++ << "]=" << w << " ; ";
                }
                out << "Inputstate=" << l.GetNeuron(i).GetInputState() << std::endl;
            }
        }
        out << "*************end of dump***************" << std::endl;

    }

    template <typename T>
    void SkyNet<T>::Serialize(std::ostream& out) {
        neural_serialization::NeuralNet net;

        net.set_layers_num(layers_.size());
        for (auto layer : layers_) {
            auto l = net.add_layers();
            l->set_neurons_num(layer.GetNeuronCount());
            for ( auto neuron : layer.GetNeurons()) {
                auto n = l->add_neurons();
                n->set_wieghts_num(neuron.weights_.size());
                n->set_id(neuron.GetId());
                n->set_input_state(neuron.GetInputState());
                for (auto weights : neuron.GetWeights()) {
                    n->add_weights(weights);
                }
            }

        }
        net.SerializeToOstream(&out);

    }


    template <typename T>
    void SkyNet<T>::DeSerialize(std::istream& in) {

        neural_serialization::NeuralNet net;
        
        net.ParseFromIstream(&in);

        layers_.clear();
        decision_.clear();
        int layers_count = 0;
        for (auto layer : net.layers()) {
            NeuronsLayer<T> l;
            for (int neuron_count = 0; neuron_count < layer.neurons_num(); neuron_count++) {
                int weight_size = layer.mutable_neurons()->Mutable(neuron_count)->wieghts_num();
                Neuron<T> n;
               
                n.inputstate_ =  layer.mutable_neurons()->Mutable(neuron_count)->input_state();
                n.SetId(layer.mutable_neurons()->Mutable(neuron_count)->id());
                auto w = layer.mutable_neurons()->Mutable(neuron_count)->weights();
                for (int i = 0; i < weight_size; i++) {
                    n.AddWeight(w.Get(i));
                }
                l.AddNeron(n);
            }
            if (layers_count == net.layers_num() - 1) {
                AddLayer(l, true); //add decision layer
            }
            else {
                AddLayer(l);
            }
            layers_count++;
        }


    }



    template <typename T>
    void SkyNet<T>::Learning(std::vector<InputSignals>& inp, std::vector<double> expected_decision, int epoche_num, double learning_rate) {

        if (inp.size() != expected_decision.size()) {
            assert("Size of vectors for learning is not equal");

        }




        for (auto x : inp) {
            if (x.signals.size() != layers_[0].GetNeuronCount()) {
                //count of input signals must be equal number of neurons in input layer
                assert("Input signals not mathing with layer1");
            }
        }

        ResetAllWeights();


        for (int epoche = 0; epoche < epoche_num; epoche++) {
            for (int i = 0; i < inp.size(); i++) {
                std::vector<double> actual_decision = GetDecision(inp[i]);

                for (int decision_num = 0; decision_num < actual_decision.size(); decision_num++) {

                    double expected;
                    if (actual_decision.size() == 1) {
                        expected = expected_decision[i];
                    }
                    else {
                        expected = expected_decision[i] == decision_num ? 1 : 0;
                    }

                    double error_decision = (actual_decision[decision_num] - expected);

                    double weights_delta = error_decision * (actual_decision[decision_num] * (1 - actual_decision[decision_num]));

                    double result_of_learning = decision_[decision_num]->Learn(-1, weights_delta, learning_rate);
                }

            }
        }

    }

    template <typename T>
    void SkyNet<T>::AddLayer(NeuronsLayer<T>& layer, bool decision_layer) {

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


        if (decision_layer) {
            for (int i = 0; i < layer.GetNeuronCount(); i++) {
                decision_.push_back(&layers_[current_layer].GetNeuron(0));
            }
        }



    }




}// end of namespace
