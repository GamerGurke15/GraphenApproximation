package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import neural_network.Neuron;

public class Main{

	public static final String INPUT_FILE = "input";
	public static final String OUTPUT_FILE = "training";
	private static final int maxNeuronsInput = 50;
	private static final int maxNeuronsHidden = 40;
	private static final int maxNeuronsOutput = 10;

	public Main(){
	}

	public static void main(String[] args){
		try{
			FileInputStream fi = new FileInputStream(new File(Main.INPUT_FILE));
			InputStreamReader reader = new InputStreamReader(fi);
			BufferedReader re = new BufferedReader(reader);
			String line;
			Neuron[] inNeurons = new Neuron[maxNeuronsInput];
			int i = 0;
			while ((line = re.readLine()) != null && i <= maxNeuronsInput){
				try{
					double tmp = Double.valueOf(line);
					Neuron ne = new Neuron();
					ne.setWert(tmp);
					inNeurons[i] = ne;
					i++;
				} catch (NumberFormatException e){
					System.err.println("Error while reading input-file in line " + i);
				}
			}
			re.close();
			reader.close();
			fi.close();
			
			fi = new FileInputStream(new File(Main.OUTPUT_FILE));
			reader = new InputStreamReader(fi);
			re = new BufferedReader(reader);
			double[] training = new double[maxNeuronsOutput];
			i = 0;
			while ((line = re.readLine()) != null && i <= maxNeuronsOutput){
				try{
					training[i] = Double.valueOf(line);
					i++;
				} catch (NumberFormatException e){
					System.err.println("Error while reading output-file in line " + i);
				}
			}
			re.close();
			reader.close();
			fi.close();
			
			Neuron[] hiddenNeurons = new Neuron[maxNeuronsHidden];
			for (i = 0; i < maxNeuronsHidden; i++){
				Neuron ne = new Neuron();
				ne.setWert(0);
				hiddenNeurons[i] = ne;
			}

			double[][] gewichteHidden = new double[maxNeuronsInput][maxNeuronsHidden];
			int j;
			for (i = 0; i < maxNeuronsInput; i++)
				for (j = 0; j < maxNeuronsHidden; j++)
					gewichteHidden[i][j] = (Math.random() * 0.5) - 0.25f;

			for (i = 0; i < maxNeuronsHidden; i++)
				hiddenNeurons[i].setWert(berechneWert(inNeurons, i, gewichteHidden));
			Neuron[] outNeurons = new Neuron[maxNeuronsOutput];
			for (i = 0; i < maxNeuronsOutput; i++){
				Neuron ne = new Neuron();
				ne.setWert(0);
				outNeurons[i] = ne;
			}
			double[][] gewichteOut = new double[maxNeuronsHidden][maxNeuronsOutput];

			for (i = 0; i < maxNeuronsHidden; i++)
				for (j = 0; j < maxNeuronsOutput; j++)
					gewichteOut[i][j] = (Math.random() * 0.5f) - 0.25;

			// lernen
			int count = 100;
			for (i=0;i<count;i++){
				System.out.println(i + "/" + count);
				lernen(inNeurons, hiddenNeurons, outNeurons, gewichteHidden, gewichteOut, training);
			}
			
			Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteHidden, gewichteOut, training);
			for (i=0;i<maxNeuronsOutput;i++)
				System.out.println(outNeurons[i].getWert());
		} catch (IOException e){
			e.printStackTrace();
		}
	}

	static double berechneWert(Neuron[] neurons, int index, double[][] gewichte){
		double summe = 0;
		int i;
		for (i = 0; i < neurons.length; i++){
			summe = neurons[i].getWert() * gewichte[i][index];
		}
		return summe;
	}

	static double Fehlerberechnung(Neuron[] inNeurons, Neuron[] hiddenNeurons, Neuron[] outNeurons,
			double[][] gewichteHidden, double[][] gewichteOut, double[] training){
		int i;
		double error = 0;
		// werte berechnen
		for (i = 0; i < maxNeuronsHidden; i++)
			hiddenNeurons[i].setWert(berechneWert(inNeurons, i, gewichteHidden));
		for (i = 0; i < maxNeuronsOutput; i++)
			outNeurons[i].setWert(berechneWert(hiddenNeurons, i, gewichteOut));
		for (i = 0; i < maxNeuronsOutput; i++){
			error += Math.pow(training[i] - outNeurons[i].getWert(), 2);
		}
		return error;
	}

	static void lernen(Neuron[] inNeurons, Neuron[] hiddenNeurons, Neuron[] outNeurons, double[][] gewichteHidden,
			double[][] gewichteOut, double[] training){
		int i, j;
		double aenderung = 0.05d;
		double lernrate = 0.5d;
		double[][] gewichteAenderungHidden = gewichteHidden.clone();
		double[][] gewichteAenderungOut = gewichteOut.clone();
		for (i = 0; i < maxNeuronsInput; i++)
			for (j = 0; j < maxNeuronsHidden; j++){
				gewichteAenderungHidden[i][j] += aenderung;
				double ableitung = (Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteAenderungHidden,
						gewichteOut, training)
						- Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteHidden, gewichteOut, training))
						/ aenderung;
				gewichteAenderungHidden[i][j] -= aenderung;
				gewichteHidden[i][j] -= lernrate * ableitung;
			}
		for (i = 0;i<maxNeuronsHidden;i++)
			for (j = 0; j < maxNeuronsOutput; j++){
				gewichteAenderungOut[i][j] += aenderung;
				double ableitung = (Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteHidden,
						gewichteAenderungOut, training)
						- Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteHidden, gewichteOut, training))
						/ aenderung;
				gewichteAenderungOut[i][j] -= aenderung;
				gewichteOut[i][j] -= lernrate * ableitung;
			}
	}

}
