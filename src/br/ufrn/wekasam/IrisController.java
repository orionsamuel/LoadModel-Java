package br.ufrn.wekasam;

import weka.core.Instance;


public class IrisController {


	private MachineLearningModel mlm;

	public Iris classifyIrisFlowe(Iris i) {
		Iris iris = i;

		String classValue = new String(classifyIrisFlower(iris));
		iris.setIrisClass(classValue);
		
		return iris;
	}
	
	private String classifyIrisFlower(Iris iris) {
		IrisUtils iu = new IrisUtils();
		
		Instance instance = iu.irisToWekaInstance(iris);
		
		iu.getDataset().add(instance);
		
		double value = -1;
		try {
			value = mlm.getCls().classifyInstance(iu.getDataset().get(0));
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//get the irisClass value
		String prediction = null; 
		prediction = new String(iu.getDataset().classAttribute().value((int)value));
		
		return prediction;

	}
	
}
