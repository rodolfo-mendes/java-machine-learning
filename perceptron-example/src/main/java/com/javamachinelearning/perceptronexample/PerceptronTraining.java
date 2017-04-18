package com.javamachinelearning.perceptronexample;

import static java.util.Arrays.copyOfRange;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.PerceptronLearning;
import org.neuroph.util.TransferFunctionType;
 
public class PerceptronTraining {
	/*
	 * 1: Iris-setosa
	 * 0: Iris-versicolor
	 */
	public static void main(String[] args) throws IOException {
		System.out.println("Loading dataset ... ");
		
		DataSet iris = loadDataSet("iris.data", 4, 1);
		
		DataSet[] datasets = iris.sample(70);
		DataSet trainingDataSet = datasets[0];
		DataSet testDataSet = datasets[1];
		
		System.out.println("Training perceptron ...");
		
		Perceptron perceptron = new Perceptron(4, 1, TransferFunctionType.STEP);
		
		PerceptronLearning perceptronLearning = new PerceptronLearning();
		perceptronLearning.setBatchMode(false);
		perceptronLearning.setLearningRate(0.1);
		perceptronLearning.setMaxError(0.01);
		perceptronLearning.setMaxIterations(100);
		
		perceptron.learn(trainingDataSet, perceptronLearning);
		
		System.out.println("Testing ...");
		
		int [][] confusionMatrix = new int[2][2];
		
		Iterator<DataSetRow> ite = testDataSet.iterator();
		while(ite.hasNext()){
			DataSetRow row = ite.next();
			
			perceptron.setInput(row.getInput());
			perceptron.calculate();
			
			int expected = (int) row.getDesiredOutput()[0];
			int predicted = (int) perceptron.getOutput()[0];
			
			confusionMatrix[expected][predicted]++;
		}
		
		System.out.println("Confusion matrix: ");
		printConfusionMatrix(confusionMatrix);
		
		System.out.println("Weigth array:");
		System.out.println(Arrays.toString(perceptron.getWeights()));
		
		System.out.println("Saving model to file 'iris.model' ...");
		perceptron.save("iris.model");
	}

	static DataSet loadDataSet(String filename, int inputs, int outputs) throws IOException{
		DataSet ds = new DataSet(inputs, outputs);
		
		try(BufferedReader br = new BufferedReader(new FileReader(filename))){
			String line = br.readLine();
			
			while(line != null){
				double [] values = new double[inputs + outputs];
				String [] record = line.split("\\s+");
				for (int i = 0; i < record.length - 1; i++) {
					values[i] = Double.valueOf(record[i]);
				}
				
				//encode categories to binary values
				if(record[record.length-1].equals("Iris-setosa")){
					values[record.length-1] = 1.0;
				}else if(record[record.length-1].equals("Iris-versicolor")){
					values[record.length-1] = 0.0;
				}
				
				ds.addRow(
					copyOfRange(values, 0, inputs),
					copyOfRange(values, inputs, inputs + outputs)
				);
				
				line = br.readLine();
			}
		}
		
		return ds;
	}
	
	static void printConfusionMatrix(int[][] confusionMatrix) {
		for (int i = 0; i < confusionMatrix.length; i++) {
			for (int j = 0; j < confusionMatrix.length; j++) {
				System.out.printf("%d ", confusionMatrix[i][j]);
			}
			System.out.println();
		}
	}
}
