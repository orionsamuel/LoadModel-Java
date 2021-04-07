package br.ufrn.wekasam;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class Reservoir {

	public static void main(String[] args) throws Exception {

		String model = new String(args[0]);
		String base = new String(args[1]);
		
		//String model = new String("src/main/resources/history_matching.model");
		//String base = new String("src/main/resources/");
		
		
		ExtensionFilter extensionFilter = new ExtensionFilter();
		
		List<File> files = new ArrayList<File>();
		files = AbrirTodos(base);
				
		MachineLearningModel mlm = new MachineLearningModel();
		mlm.loadModel(model);

		Instances instances;
		
		for(File f: files) {
			if(extensionFilter.accept(f)) {
				instances = DataSource.read(f.getAbsolutePath());
				instances.setClassIndex(instances.numAttributes() - 1);
				System.out.println();
				System.out.println(instances.relationName());
				System.out.println();
				
				
				Instances newInstances = new Instances(instances);
				newInstances.clear();
				
				for(Instance i: instances) {
					double d = mlm.getCls().classifyInstance(i);
					
					Instance newInstance = i;
					newInstance.setClassValue(d);
					newInstances.add(newInstance);
				}
				
				outPutFile(newInstances);
				
			}
		}
		
	}
	
	private static List<File> AbrirTodos(String urlBase) {

		List<File> filess = new ArrayList<File>();
		String url = urlBase;
		File f = new File(url);
		File files[];
		files = f.listFiles();
		
		for(int i = 0; i < files.length; i++) {
			filess.add(files[i]);
		}
		return filess;
	}
	
	private static void outPutFile(Instances instances) throws IOException {
		CSVSaver saver = new CSVSaver();
	    saver.setInstances(instances);
	    saver.setMaxDecimalPlaces(30);
	    saver.setFile(new File(instances.relationName() + "_output.csv"));
	    saver.writeBatch();
	}

}
