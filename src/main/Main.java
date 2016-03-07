package main;

import neural_network.Neuron;

import java.io.*;
import java.util.Date;

public class Main{

  public static final String INPUT_FILE = "input";
  public static final String OUTPUT_FILE = "training";
  public static final String TEST_FILE = "test";
  private static final int maxNeuronsInput = 50;
  private static final int maxNeuronsHidden = 40;
  private static final int maxNeuronsOutput = 10;


  private static final int iterations = 1000;

  public Main(){
  }

  public static void main(String[] args){
    try{
      long t_0 = new Date().getTime();
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
        }catch (NumberFormatException e){
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
        }catch (NumberFormatException e){
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
          gewichteHidden[i][j] = Math.random() - 0.5f;

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
          gewichteOut[i][j] = (Math.random()) - 0.5f;

      for (i = 0; i < iterations; i++){
        if (i % 100 == 0) System.out.println(i + "/" + iterations);
        lernen(inNeurons, hiddenNeurons, outNeurons, gewichteHidden, gewichteOut, training);
      }
      Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteHidden, gewichteOut, training);
      for (i = 0; i < maxNeuronsOutput; i++)
        System.out.println(outNeurons[i].getWert());

      System.out.println("dt=" + (-(t_0 - new Date().getTime()) / 1000) + "s");
    }catch (IOException e){
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
    double aenderung = 0.01d;
    double lernrate = 0.00001d;
    double[][] gewichteAenderungHidden = new double[maxNeuronsInput][maxNeuronsHidden];
    double[][] gewichteAenderungOut = new double[maxNeuronsHidden][maxNeuronsOutput];
    for (i = 0; i < maxNeuronsInput; i++)
      for (j = 0; j < maxNeuronsHidden; j++){
        gewichteAenderungHidden[i][j] = gewichteHidden[i][j];
      }
    for (i = 0; i < maxNeuronsHidden; i++)
      for (j = 0; j < maxNeuronsOutput; j++){
        gewichteAenderungOut[i][j] = gewichteOut[i][j];
      }
    for (i = 0; i < maxNeuronsInput; i++)
      for (j = 0; j < maxNeuronsHidden; j++){
        gewichteHidden[i][j] = gewichteHidden[i][j] + aenderung;
        double ableitung = ((Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteHidden, gewichteOut,
            training)
            - (Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteAenderungHidden, gewichteOut,
            training))))
            / aenderung;
        gewichteHidden[i][j] -= aenderung;
        gewichteHidden[i][j] -= lernrate * ableitung;
      }
    for (i = 0; i < maxNeuronsHidden; i++)
      for (j = 0; j < maxNeuronsOutput; j++){
        gewichteOut[i][j] += aenderung;
        double ableitung = ((Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteHidden, gewichteOut,
            training)
            - Fehlerberechnung(inNeurons, hiddenNeurons, outNeurons, gewichteHidden, gewichteAenderungOut,
            training)))
            / aenderung;
        gewichteOut[i][j] -= aenderung;
        gewichteOut[i][j] -= lernrate * ableitung;
      }

  }

}
