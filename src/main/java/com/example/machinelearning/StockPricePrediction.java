package com.example.machinelearning;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class StockPricePrediction {

  public static void main(String[] args) throws Exception {
    // Krijimi i atributit për çmimin e aksionit
    Attribute priceAttribute = new Attribute("price");

    // Krijimi i listës së atributeve
    java.util.ArrayList<Attribute> attributeList = new java.util.ArrayList<Attribute>(1);
    attributeList.add(priceAttribute);

    // Krijimi i një instance të zbrazët
    Instances dataset = new Instances("stock_dataset", attributeList, 0);

    // Vendosja e klases për të parashikuar (çmimi i aksionit)
    dataset.setClassIndex(0);

    // Krijimi i një instance të re me vlera të dhëna për parashikim
    Instance newInst = new DenseInstance(1);
    newInst.setValue(priceAttribute, 150.0); // Vlera e dhënë e shembullit

    // Shtimi i instance në dataset
    dataset.add(newInst);

    // Krijimi i një modelet të mësimit të marrjes së vendimit (Linear Regression)
    LinearRegression model = new LinearRegression();
    model.buildClassifier(dataset);

    // Bërja e parashikimit për vlerën e aksionit
    double predictedPrice = model.classifyInstance(newInst);

    System.out.println("Parashikimi i çmimit të aksionit: " + predictedPrice);
  }
}

