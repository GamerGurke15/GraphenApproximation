package neural_network;

import java.util.ArrayList;

public class Neuron{
	
	private ArrayList<Double> Gewicht;
	private double wert;
	
	public Neuron(){
		Gewicht = new ArrayList<>();
		wert = -1;
	}
	
	public ArrayList<Double> getGewicht(){
		return Gewicht;
	}
	
	public double getWert(){
		return wert;
	}
	
	public void setGewichte(ArrayList<Double> ge){
		Gewicht = duplicateList(ge);
	}

	public void setWert(double tmp){
		wert = tmp;
	}
	
	
	private ArrayList<Double> duplicateList(ArrayList<Double> ge) {
		ArrayList<Double> dbl = new ArrayList<>();
		for (int i=0;i<ge.size();i++){
			Double tmp = new Double(ge.get(i));
			dbl.add(tmp);
		}
		return dbl;
	}
}