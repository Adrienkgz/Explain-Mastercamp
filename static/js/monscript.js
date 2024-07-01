function afficherMessage(){
    // Possibilité de tester dans la console
    console.log("hello");

    // Récupération des valeurs des champs remplis par l'utilisateur 
    var nom = document.forms.monformulaire.elements.field1.value;
    var prenom = document.forms.monformulaire.elements.field2.value;
    var mail = document.forms.monformulaire.elements.field3.value;
    var message = document.forms.monformulaire.elements.field5.value;
    
    // Message complet que je renvoie à l'utilisateur à partir des données qu'il me renvoie
    var renvoieMessage = "Bienvenue " + prenom + " " + nom + ", je suis ravie de faire votre connaissance. Voici votre message : " + message + ". Merci à vous.";
    
    //Boite de dialogue pour indiquer que le message a bien été enregistrée
    alert("Hop ! J'ai bien intercepté votre message ! \n Vous receverez une réponse dans quelques heures sur : " + mail);

}