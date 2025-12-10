class Node {
  int data;
  Node? left;
  Node? right;

  Node(this.data, [this.left, this.right]);

  void displayAll(Node? root) {
    if (root == null) {
      return;
    }
    displayAll(root.left);
    print(root.data);
    displayAll(root.right);
  }
}

class RootNode extends Node {
  // RootNode(super.data); // more modern version
  RootNode(int data) : super(data);

  Node insert(int data, Node? node){
    if(node == null){
      return Node(data);
    }
    if(node.data > data){
      node.left = insert(data, node.left);
    }
    if(node.data < data){
      node.right = insert(data, node.right);
    }
    return node;
  }
}

void main(){
  RootNode root = RootNode(2);
  root.insert(3, root);
  root.insert(1, root);
  // Akan return node yang memiliki nomor yang sama jadi tidak akan ada duplikasi.
  root.insert(1, root);
  root.displayAll(root);
}