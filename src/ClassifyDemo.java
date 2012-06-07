import java.io.File;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class ClassifyDemo{
	
	public static void main(String[] args){
		//String path = "C://Program Files//Weka-3-6//data//iris.arff"; //数据集路径
		String path = "D://10y~.arff";
		
		Classifier cfs;
		
		try{
		
			//读入样本
			File file = new File(path);
			ArffLoader loader = new ArffLoader();
			loader.setFile(file);
		
			//数据集载入样本
			Instances dataSet = loader.getDataSet();
		
			/*在使用样本之前一定要首先设置instances的classIndex，否则在使用instances对象时会抛出异常
			Sets the class index of the set. 应该是确定分类的目标层*/
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
			//初始BP神经网络 以MultilayerPerceptron算法实现分类器
			cfs = new MultilayerPerceptron();
		
			//使用样本训练分类器
			cfs.buildClassifier(dataSet);
			
			//获取需要测试的demo
			Instance demo = new Instance(dataSet.numAttributes());
			demo.setDataset(dataSet);
			
			//模拟demo的数据
			double[] Data = { 120, 121, 122, 123, 124, 125, 126, 127, 128 };
			//double[] Data = { 5.1,3.5,1.4,0.2 };
			for (int i = 0; i < demo.numAttributes() - 1; i++) {
				demo.setValue(i, Data[i]);
			}
			
			//run demo!
			double valIndex = cfs.classifyInstance(demo);
			
			//WTF!
			Attribute attr = dataSet.attribute(dataSet.classIndex());
			System.out.println("valIndex: " + valIndex );
			System.out.println("Result: " + attr.value( (int)valIndex ));
			
		}catch(Exception e){
			e.printStackTrace();
		}
	}

}