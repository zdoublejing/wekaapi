import java.io.File;

import weka.clusterers.SimpleKMeans;
//import weka.core.DistanceFunction;
//import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
 
public class SimpleCluster {

    /**
     * @param args
     */

    public static void main(String[] args) {

       // TODO Auto-generated method stub

       Instances ins = null;
       Instances tempIns = null;

       SimpleKMeans KM = null;
       //DistanceFunction disFun = null;

       try{
           /*
            * 1.读入样本
            */
           File file= new File("C://Program Files//Weka-3-6//data//contact-lenses.arff");
           ArffLoader loader = new ArffLoader();
           loader.setFile(file);
           ins = loader.getDataSet();
        
           /*
            * 2.初始化聚类器
            * 在3.6版本可以通过setDistanceFunction(DistanceFunction df)
            * 函数设置聚类算法内部的距离计算方式
            * 而在3.5版本里面默认的采用了欧几里得距离
            */
           KM = new SimpleKMeans();       

           //设置聚类要得到的类别数量
           KM.setNumClusters(2);
        
           /*
            * 3.使用聚类算法对样本进行聚类
            */
           KM.buildClusterer(ins);          

           /*
            * 4.打印聚类结果
            */
           tempIns = KM.getClusterCentroids();
           System.out.println("CentroIds: " + tempIns);

       }catch(Exception e){

           e.printStackTrace();

       }
    }
    
}