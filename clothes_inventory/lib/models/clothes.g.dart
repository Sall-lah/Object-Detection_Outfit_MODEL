// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'clothes.dart';

// **************************************************************************
// TypeAdapterGenerator
// **************************************************************************

class ClothesAdapter extends TypeAdapter<Clothes> {
  @override
  final int typeId = 0;

  @override
  Clothes read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{
      for (int i = 0; i < numOfFields; i++) reader.readByte(): reader.read(),
    };
    return Clothes(
      name: fields[0] as String,
      color: fields[1] as String,
      imagePath: fields[2] as String,
    );
  }

  @override
  void write(BinaryWriter writer, Clothes obj) {
    writer
      ..writeByte(3)
      ..writeByte(0)
      ..write(obj.name)
      ..writeByte(1)
      ..write(obj.color)
      ..writeByte(2)
      ..write(obj.imagePath);
  }

  @override
  int get hashCode => typeId.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ClothesAdapter &&
          runtimeType == other.runtimeType &&
          typeId == other.typeId;
}
