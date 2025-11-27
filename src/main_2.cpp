#include <iostream>
#include <algorithm>
#include <cmath>
#include "FHEController.h"
#include <chrono>
#include <filesystem>

#define GREEN_TEXT "\033[1;32m"
#define RED_TEXT "\033[1;31m"

using namespace std::chrono;
namespace fs = std::filesystem;

enum class Parameters { Generate, Load };

FHEController controller;

Ctxt encoder1();
Ctxt pooler(Ctxt input);
Ctxt classifier(Ctxt input);

string input_folder;

string text;

bool verbose = false;
Parameters p = Parameters::Load;
bool security128bits = false;

double read_value(const string &filename) {
    ifstream fin(filename);
    double val;
    if (!(fin >> val)) {
        cerr << "读取 " << filename << " 失败" << endl;
        exit(1);
    }
    return val;
}

void setup_environment(int argc, char *argv[]) {
    string command;

    if (argc < 2) {
        cout << "This is FHE-Linformer, an encrypted text classifier based on Transformer.\n\nUsage: ./FHE-Linformer <text_input> [OPTIONS]\n\nthe following [OPTIONS] are available:\n--verbose: activates verbose mode\n--secure: creates parameters with 128 bits of security. Use only if necessary, as it adds computational overhead \n\nExample:\n./FHE-Linformer --verbose\n";
        exit(0);
    } else {
        if (string(argv[1]) == "--generate_keys")
        {
            if (argc > 2 && string(argv[2]) == "--secure") {
                security128bits = true;
            }

            p = Parameters::Generate;
            return;
        }

        for (int i = 1; i < argc; i++) {
            if (string(argv[i]) == "--verbose") {
                verbose = true;
            }
        }

        // if (verbose) cout << "\nCLIENT-SIDE\nTokenizing..." << endl;
        // command = "python3 ../src/python/ExtractEmbeddings.py";
        // system(command.c_str());
        // command = "python3 ../src/python/extract_parameters_numeric.py";
        // system(command.c_str());
        // command = "python3 ../src/python/dimReduce.py";
        // system(command.c_str());
        // command = "python3 ../src/python/mergeweights.py"
        // system(command.c_str());

    }

}

int main(int argc, char* argv[]) {
    input_folder = "/Users/tangxianning/Downloads/FHE-Linformer/src/tmp_embeddings/test_0/";
    setup_environment(argc, argv);

    if (p == Parameters::Generate) {
        system("mkdir -p ../keys");
        controller.generate_context(true, security128bits);
        vector<int> rotations = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, -1, -2, -4, -8 -16, -32, -64};
        controller.generate_bootstrapping_and_rotation_keys(rotations, 16384, true, "rotation_keys.txt");
        return 0;
    } else if (p == Parameters::Load) {
        controller.load_context(false);
        controller.load_bootstrapping_and_rotation_keys("rotation_keys.txt", 16384, false);
    }

    system("mkdir -p ../checkpoint");

    if(verbose) cout << "\nSERVER-SIDE\nThe evaluation of the circuit started." << endl;

    auto start = high_resolution_clock::now();

    if(input_folder.empty()) {
        cerr << "The input folder \"" << input_folder << "\" is empty!";
        exit(1);
    }

    Ctxt encoder1output;

    encoder1output = encoder1();

    // encoder1output = controller.load_ciphertext("../checkpoint/encodered.bin");
    Ctxt pooled = pooler(encoder1output);

    Ctxt classified = classifier(pooled);

    if (verbose) cout << "The circuit has been evaluated, the results are sent back to the client" << endl << endl;
    if (verbose) cout << "CLIENT-SIDE" << endl;

    vector<double> plain_result = controller.decrypt_tovector(classified, 16384);

    int timing = (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0;
    if (verbose) cout << endl << "The evaluation of the FHE circuit took: " << timing << " seconds." << endl;

    vector<double> softmax_test;
    for (int i = 0; i < 20; i++) {
        softmax_test.push_back(plain_result[i*128]);
    }

    double maxv = *max_element(softmax_test.begin(), softmax_test.end());
    vector<double> softmax_prob;
    softmax_prob.reserve(softmax_test.size());
    double sumexp = 0.0;
    for (int i = 0; i < softmax_test.size(); i++) {
        double v = exp(softmax_test[i] - maxv);
        softmax_prob.push_back(v);
        sumexp += v;
    }
    for (int i = 0; i < softmax_prob.size(); i++) {
        softmax_prob[i] /= sumexp;
    }
    int pred = (int)distance(softmax_prob.begin(), max_element(softmax_prob.begin(), softmax_prob.end()));
    
    for (int i = 0; i < softmax_prob.size(); i++) {
        cout << "Softmax Prob: " << softmax_prob[i] << endl;
    }
    cout << "Pred: " << pred << endl;
}

Ctxt encoder1() {
    // self-attention
    int inputs_count = 1;

    std::filesystem::path p1 { input_folder };

    for (__attribute__((unused)) auto& p : std::filesystem::directory_iterator(p1))
    {
        ++inputs_count;
    }

    if (verbose) cout << inputs_count << " inputs found!" << endl << endl;

    
    vector<Ctxt> inputs_E;
    for (int i = 0; i < 32; i++) {
        inputs_E.push_back(controller.read_expanded_input("../input/XE_" + to_string(i) + ".txt")); // 32 * 128
    }

    vector<Ctxt> inputs_F;
    for (int i = 0; i < 32; i++) {
        inputs_F.push_back(controller.read_expanded_input("../input/XF_" + to_string(i) + ".txt")); // 32 * 128
    }

    vector<Ctxt> inputs;
    inputs.push_back(controller.read_expanded_input("../weights-20NG/cls_token.txt"));
    for (int i = 0; i < inputs_count - 1; i++) {
        inputs.push_back(controller.read_expanded_input(input_folder + "input_" + to_string(i) + ".txt"));
    }

    auto start = high_resolution_clock::now();

    Ptxt query_w = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WQ_weight_T.txt");
    Ptxt query_b = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WQ_bias.txt");
    Ptxt key_w = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WK_weight_T.txt");
    Ptxt key_b = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WK_bias.txt");

    vector<Ctxt> Q = controller.matmulRE(inputs, query_w, query_b);
    vector<Ctxt> K = controller.matmulRE(inputs_E, key_w, key_b);

    Ctxt K_wrapped = controller.wrapUpRepeated(K);

    vector<Ctxt> Q_1;
    vector<Ctxt> Q_2;
    for (int i = 0; i < 128; i++) {
        Q_1.push_back(Q[i]);
    }
    for (int i = 128; i < Q.size(); i++) {
        Q_2.push_back(Q[i]);
    }

    Ctxt scores_1 = controller.matmulScores(Q_1, K_wrapped);
    scores_1 = controller.eval_exp(scores_1, Q_1.size());

    Ctxt scores_2 = controller.matmulScores(Q_2, K_wrapped);
    scores_2 = controller.eval_exp(scores_2, Q_2.size());

    Ctxt scores_sum_1 = controller.rotsum(scores_1, 32, 128);
    Ctxt scores_sum_2 = controller.rotsum(scores_2, 32, 128);
    controller.print(scores_sum_1, 128, "scores_sum_1");
    controller.print_padded(scores_sum_1, 32, 128, "scores_sum_1_padded");

    controller.print_min_max(scores_sum_1);
    controller.print_min_max(scores_sum_2);

    Ctxt scores_denominator_1 = controller.eval_inverse_naive(scores_sum_1, -1, 190000);
    Ctxt scores_denominator_2 = controller.eval_inverse_naive(scores_sum_2, -1, 190000);

    scores_1 = controller.mult(scores_1, scores_denominator_1);
    scores_2 = controller.mult(scores_2, scores_denominator_2);

    vector<Ctxt> unwrapped_scores_1 = controller.unwrapExpanded(scores_1, 128);
    vector<Ctxt> unwrapped_scores_2 = controller.unwrapExpanded(scores_2, inputs.size() - 128);

    vector<Ctxt> unwrapped_scores;
    unwrapped_scores.insert(unwrapped_scores.end(), unwrapped_scores_1.begin(), unwrapped_scores_1.end());
    unwrapped_scores.insert(unwrapped_scores.end(), unwrapped_scores_2.begin(), unwrapped_scores_2.end());

    Ptxt value_w = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WV_weight_T.txt");
    Ptxt value_b = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WV_bias.txt");

    vector<Ctxt> V = controller.matmulRE(inputs_F, value_w, value_b);
    Ctxt V_wrapped = controller.wrapUpRepeated(V);

    vector<Ctxt> output = controller.matmulRE(unwrapped_scores, V_wrapped, 128, 128);

    if (verbose) cout << "The evaluation of Self-Attention took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print(output[0], 128, "Self-Attention (Repeated)");

    // 残差 + Affine1    

    start = high_resolution_clock::now();

    Ptxt dense_w = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WO_weight.txt", output[0]->GetLevel());
    Ptxt dense_b = controller.read_plain_expanded_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WO_bias.txt", output[0]->GetLevel() + 1);

    output = controller.matmulCR(output, dense_w, dense_b);

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.add(output[i], inputs[i]);
    }

    double c10 = read_value("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_c0.txt");
    double c11 = read_value("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_c1.txt");
    double c12 = read_value("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_c2.txt");

    size_t S = output.size();
    double fL1 = c10 + c11 / sqrt(S) + c12 / S;

    vector<Ctxt> output_0;
    vector<Ctxt> output_1;
    for (int i = 0;i < 128; i++) {
        output_0.push_back(output[i]);
    }
    for (int i = 128; i < output.size(); i++) {
        output_1.push_back(output[i]);
    }

    Ctxt wrappedOutput_0 = controller.wrapUpExpanded(output_0);
    Ctxt wrappedOutput_1 = controller.wrapUpExpanded(output_1);

    Ptxt a1 = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_a.txt", wrappedOutput_0->GetLevel(), fL1);
    Ptxt b1 = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_b.txt", wrappedOutput_0->GetLevel() + 1, fL1);

    wrappedOutput_0 = controller.mult(wrappedOutput_0, a1);
    wrappedOutput_0 = controller.add(wrappedOutput_0, b1);
    wrappedOutput_1 = controller.mult(wrappedOutput_1, a1);
    wrappedOutput_1 = controller.add(wrappedOutput_1, b1);

    wrappedOutput_0 = controller.bootstrap(wrappedOutput_0);
    wrappedOutput_1 = controller.bootstrap(wrappedOutput_1);

    Ctxt output_copy_0 = wrappedOutput_0->Clone();
    Ctxt output_copy_1 = wrappedOutput_1->Clone();

    output_0 = controller.unwrapExpanded(wrappedOutput_0, 128);
    output_1 = controller.unwrapExpanded(wrappedOutput_1, inputs.size() - 128);

    if (verbose) cout << "The evaluation of Self-Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print_expanded(output_0[0], 0, 128, "Self-Output (Expanded)");

    // FFN

    start = high_resolution_clock::now();

    double GELU_max_abs_value = 1.0 / 8.0;

    // 这部分检查一下
    Ptxt intermediate_w_1 = controller.read_plain_input("../weights-20NG/ffn_W0_transposed_block_0.txt", wrappedOutput_0->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_2 = controller.read_plain_input("../weights-20NG/ffn_W0_transposed_block_1.txt", wrappedOutput_0->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_3 = controller.read_plain_input("../weights-20NG/ffn_W0_transposed_block_2.txt", wrappedOutput_0->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_4 = controller.read_plain_input("../weights-20NG/ffn_W0_transposed_block_3.txt", wrappedOutput_0->GetLevel(), GELU_max_abs_value);

    vector<Ptxt> dense_weights = {intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4};

    Ptxt intermediate_bias = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_ffn_Wffn_0_bias.txt", wrappedOutput_0->GetLevel() + 1, GELU_max_abs_value);

    output_0 = controller.matmulRElarge(output_0, dense_weights, intermediate_bias);
    output_1 = controller.matmulRElarge(output_1, dense_weights, intermediate_bias);

    // 合并两个输出向量
    vector<Ctxt> outputs_raw;
    outputs_raw.reserve(output_0.size() + output_1.size());
    outputs_raw.insert(outputs_raw.end(), output_0.begin(), output_0.end());
    outputs_raw.insert(outputs_raw.end(), output_1.begin(), output_1.end());
    cout << outputs_raw.size() << endl;

    // 统一打包为容器密文
    auto outputs = controller.generate_containers(outputs_raw, nullptr);

    for (int i = 0; i < outputs.size(); i++) {
        controller.print_min_max(outputs[i]);
        outputs[i] = controller.eval_gelu_function(outputs[i], -1, 1, GELU_max_abs_value, 119);
        outputs[i] = controller.bootstrap(outputs[i]);
    }

    vector<vector<Ctxt>> unwrappedLargeOutput = controller.unwrapRepeatedLarge(outputs, inputs.size());

    if (verbose) cout << "The evaluation of Intermediate took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print(unwrappedLargeOutput[0][0], 128, "Intermediate (Containers)");

    // 残差 + Affine2

    Ptxt output_w_1 = controller.read_plain_input("../weights-20NG/ffn_W2_block_0.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_2 = controller.read_plain_input("../weights-20NG/ffn_W2_block_1.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_3 = controller.read_plain_input("../weights-20NG/ffn_W2_block_2.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_4 = controller.read_plain_input("../weights-20NG/ffn_W2_block_3.txt", unwrappedLargeOutput[0][0]->GetLevel());

    Ptxt output_bias = controller.read_plain_expanded_input("../weights-20NG/linformer_transformerLayers_transformer0_ffn_Wffn_2_bias.txt", unwrappedLargeOutput[0][0]->GetLevel() + 1);

    output = controller.matmulCRlarge(unwrappedLargeOutput, {output_w_1, output_w_2, output_w_3, output_w_4}, output_bias);

    vector<Ctxt> output_2;
    vector<Ctxt> output_3;

    for (int i = 0; i < 128; i++) {
        output_2.push_back(output[i]);
    }
    for (int i = 128; i < output.size(); i++) {
        output_3.push_back(output[i]);
    }

    Ctxt wrappedOutput_2 = controller.wrapUpExpanded(output_2);
    Ctxt wrappedOutput_3 = controller.wrapUpExpanded(output_3);

    wrappedOutput_2 = controller.add(wrappedOutput_2, output_copy_0);
    wrappedOutput_3 = controller.add(wrappedOutput_3, output_copy_1);

    double c20 = read_value("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine2_c0.txt");
    double c21 = read_value("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine2_c1.txt");
    double c22 = read_value("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine2_c2.txt");

    size_t S_1 = output.size();
    double fL2 = c20 + c21 / sqrt(S_1) + c22 / S_1;

    cout << "fL2: " << fL2 << endl;

    Ptxt a2 = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine2_a.txt", wrappedOutput_2->GetLevel(), fL2);
    Ptxt b2 = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine2_b.txt", wrappedOutput_3->GetLevel() + 1, fL2);

    wrappedOutput_2 = controller.mult(wrappedOutput_2, a2);
    wrappedOutput_3 = controller.mult(wrappedOutput_3, a2);

    wrappedOutput_2 = controller.add(wrappedOutput_2, b2);
    wrappedOutput_3 = controller.add(wrappedOutput_3, b2);

    output_2 = controller.unwrapExpanded(wrappedOutput_2, 128);
    output_3 = controller.unwrapExpanded(wrappedOutput_3, inputs.size() - 128);

    if (verbose) cout << "The evaluation of Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print_expanded(output_2[0], 0, 128,"Output (Expanded)");
    
    controller.save(output_2[0], "../checkpoint/encodered.bin");

    return output_2[0];
}

Ctxt pooler(Ctxt input) {
    auto start = high_resolution_clock::now();

    double tanhScale = 1.0 / 18.0;

    Ptxt weight = controller.read_plain_input("../weights-20NG/pooler_dense_weight_T.txt", input->GetLevel(), tanhScale);
    Ptxt bias = controller.read_plain_repeated_input("../weights-20NG/pooler_dense_bias.txt", input->GetLevel() + 1, tanhScale);

    Ctxt output = controller.mult(input, weight);

    output = controller.rotsum(output, 128, 128);

    output = controller.add(output, bias);

    output = controller.bootstrap(output);

    controller.print_min_max(output);

    output = controller.eval_tanh_function(output, -1, 1, tanhScale, 300);

    if (verbose) cout << "The evaluation of Pooler took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print(output, 128, "Pooler (Repeated)");

    return output;
}

Ctxt classifier(Ctxt input) {
    Ptxt weight = controller.read_plain_input("../weights-20NG/fcLinear_0_weight.txt", input->GetLevel());
    Ptxt bias = controller.read_plain_expanded_input("../weights-20NG/fcLinear_0_bias.txt", input->GetLevel());

    Ctxt output = controller.mult(input, weight);

    output = controller.rotsum(output, 128, 1);

    output = controller.add(output, bias);

    vector<double> mask;
    for (int i = 0; i < controller.num_slots; i++) {
        mask.push_back(0);
    }

    for (int i = 0; i < 20; i++) {
        mask[i * 128] = 1;
    }

    output = controller.mult(output, controller.encode(mask, output->GetLevel(), controller.num_slots));

    return output;
}
