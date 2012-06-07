import java.io.File;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class demo{
	
	public static void main(String[] args){
		
		Instances ins = null;
		MultilayerPerceptron BP = null;
		
		try{
			//读入样本
			File file = new File("C://Program Files//Weka-3-6//data//iris.arff");
			ArffLoader loader = new ArffLoader();
			loader.setFile(file);
			ins = loader.getDataSet();
			
			//在使用样本之前一定要首先设置instances的classIndex，否则在使用instances对象是会抛出异常
			ins.setClassIndex(ins.numAttributes() - 1);
			
			//初始BP神经网络 以MultilayerPerceptron算法实现 分类器
			BP = new MultilayerPerceptron();
			
			//使用样本训练分类器
			BP.buildClassifier(ins);
			
			//BP.classifyInstance(arg0); 预测分类一个样本
			
			//检验分类器的评估器
			Evaluation eval= new Evaluation(ins);
			
			//使用同样的样本进行测试
			Instance test;
			
			//用每个样本评估分类器
			for(int i = 0; i < ins.numInstances(); i++) {
				test = ins.instance(i);
				eval.evaluateModelOnceAndRecordPrediction(BP, test);
			}
			
			//打印结果
			System.out.println("分类器的正确率：" + (1- eval.errorRate()));
			
		}catch(Exception e){
			e.printStackTrace();
		}
		
	}
}