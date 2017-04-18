package com.javamachinelearning.perceptronexample;

import java.util.Arrays;

import org.neuroph.nnet.Perceptron;

public class PerceptionInProduction {
	public static void main(String[] args) {
		Perceptron perceptron = (Perceptron) Perceptron.createFromFile("iris.model");
		
		double [][] testData = {
			{4.6, 3.2, 1.4, 0.2},
			{5.3, 3.7, 1.5, 0.2},
			{5.0, 3.3, 1.4, 0.2},
			{7.0, 3.2, 4.7, 1.4},
			{6.4, 3.2, 4.5, 1.5},
			{6.9, 3.1, 4.9, 1.5},
		};
		
		for (int i = 0; i < testData.length; i++) {
			perceptron.setInput(testData[i]);
			perceptron.calculate();
			int out = (int) perceptron.getOutput()[0];
			
			String category = out > 0 ? "Iris-setosa" : "Iris-versicolor";
			
			System.out.printf("%s | %s%n", Arrays.toString(testData[i]), category);
		}
	}
}
