package com.example.machinelearning;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

public class StockPricePrediction extends Application {

  @Override
  public void start(Stage primaryStage) throws Exception {
    // Krijimi i atributit për çmimin e aksionit
    Attribute priceAttribute = new Attribute("price");

    // Krijimi i listës së atributeve
    ArrayList<Attribute> attributeList = new ArrayList<>();
    attributeList.add(priceAttribute);

    // Krijimi i një instance të zbrazët
    Instances dataset = new Instances("stock_dataset", attributeList, 0);

    // Vendosja e klases për të parashikuar (çmimi i aksionit)
    dataset.setClassIndex(0);

    // Krijimi i një TextField për të lejuar përdoruesin të vendosë vlerën
    TextField priceInput = new TextField();
    priceInput.setPromptText("Vendosni çmimin e aksionit");

    // Krijimi i një Label për treguar parashikimin
    Label predictionLabel = new Label();

    // Krijimi i një skene
    VBox root = new VBox(priceInput, predictionLabel);
    Scene scene = new Scene(root, 300, 200);

    // Vendosja e skenës në dritaren kryesore
    primaryStage.setScene(scene);
    primaryStage.setTitle("Stock Price Prediction");
    primaryStage.show();

    // Vendosja e një dëgjuesi për të reaguar kur përdoruesi shtyp Enter në TextField
    priceInput.setOnAction(event -> {
      try {
        double inputValue = Double.parseDouble(priceInput.getText());

        // Krijimi i një instance të re me vlerat e dhëna nga TextField
        Instance newInst = new DenseInstance(1);
        newInst.setValue(priceAttribute, inputValue);

        // Shtimi i instance në dataset
        dataset.add(newInst);

        // Krijimi i një modeli të mësimit të marrjes së vendimit (Linear Regression)
        LinearRegression model = new LinearRegression();
        model.buildClassifier(dataset);

        // Bërja e parashikimit për vlerën e aksionit
        double predictedPrice = model.classifyInstance(newInst);

        // Përditësimi i tekstit të Label me parashikimin
        predictionLabel.setText("Parashikimi i çmimit të aksionit: " + predictedPrice);
      } catch (NumberFormatException e) {
        predictionLabel.setText("Vendosni një vlerë të vlefshme numerike.");
      } catch (Exception e) {
        predictionLabel.setText("Gabim gjatë parashikimit.");
      }
    });
  }

  public static void main(String[] args) {
    launch(args);
  }
}
