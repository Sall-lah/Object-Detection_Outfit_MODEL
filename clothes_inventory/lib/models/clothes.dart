import 'package:hive/hive.dart';

part 'clothes.g.dart';

@HiveType(typeId: 0)
class Clothes {
  @HiveField(0)
  String name;

  @HiveField(1)
  String color;

  @HiveField(2)
  String imagePath;

  Clothes({required this.name, required this.color, required this.imagePath});
}
