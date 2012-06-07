import java.io.File;
import java.io.IOException;

import weka.classifiers.*;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * @author Jia Yu
 * @date 2010-6-9
 * http://www.blogjava.net/changedi/archive/2010/11/04/337197.html
 */

public class Classifier {
	/**
	 *@param args
	 */
	private ArffLoader loader;
	private Instances dataSet;
	private File arffFile;
	private int sizeOfDataset;
	private Classifier classifier;
	private int sizeOfAttribute;
	private String resultClass; //mark
	private double[] distributions;
	
	public Classifier(File file) throws Exception {
		this.arffFile = file;
		loadTrainSet();
		loadClassifier();
		buildClassifierModel();
	}
	
	private void buildClassifierModel() throws Exception {
		this.classifier.buildClassifier(dataSet);
	}
	
	private void loadClassifier() {
		this.classifier = new weka.classifiers.RandomSubSpace();
	}
	
	private void loadTrainSet() throws IOException {
		loader = new ArffLoader();
		loader.setFile(this.arffFile);
		dataSet = loader.getDataSet();
		setSizeOfDataset(dataSet.numInstances());
		setSizeOfAttribute(dataSet.numAttributes());
		dataSet.setClassIndex(this.sizeOfAttribute - 1);
	}
	
	public void classifyInstance(weka.core.Instance instance) throws Exception {
		double tNum = this.classifier.classifyInstance(instance);
		setDistributions(this.classifier.distributionForInstance(instance));
		Attribute attr = dataSet.attribute(dataSet.classIndex()); //mark
		int classIndex = (int) tNum;
		setResultClass(attr.value(classIndex)); //mark
	}
	
	public static void main(String[] args) {
		File file = new File("C://Program Files//Weka-3-6//data//iris.arff");
		
		try {
			Classifier wc = new Classifier(file);
			double[] feature = { 5.1,3.5,1.4,0.2 }; //mark
			weka.core.Instance ins = new weka.core.Instance(wc.getSizeOfAttribute());
			ins.setDataset(wc.getDataSet());
			for (int i = 0; i < ins.numAttributes() - 1; i++) {
				ins.setValue(i, feature[i]);
				// System.out.println(ins.attribute(i).getLowerNumericBound());
			}
			ins.setValue(ins.numAttributes() - 1, "Iris-setosa");
			System.out.println("original class is " + ins.stringValue(ins.numAttributes() - 1)); //mark
			wc.classifyInstance(ins);
			System.out.println("classify it to class " + wc.getResultClass()); //mark
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public int getSizeOfAttribute() {
		return sizeOfAttribute;
	}
	
	public void setSizeOfAttribute(int sizeOfAttribute) {
		this.sizeOfAttribute = sizeOfAttribute;
	}

    public Instances getDataSet() {
    	return dataSet;
    }
    
    public void setDataSet(Instances dataSet) {
    	this.dataSet = dataSet;
    }
    
    public String getResultClass() { //mark
    	return resultClass;
    }
    
    public void setResultClass(String resultClass) {
    	this.resultClass = resultClass;
    }

    public void setDistributions(double[] distributions) {
    	this.distributions = distributions;
    	}

    public double[] getDistributions() {
        return distributions;
    }

    public void setSizeOfDataset(int sizeOfDataset) {
        this.sizeOfDataset = sizeOfDataset;
    }

    public int getSizeOfDataset() {
        return sizeOfDataset;
    }
}
