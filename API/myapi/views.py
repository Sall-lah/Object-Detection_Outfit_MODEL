from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import user
from .serializer import UserSerializer

# END POINT
@api_view(['GET'])
def get_users(request):
    try:
        # Ambil semua data dari DB
        users = user.objects.all()
        # Ubah data jadi JSON
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except user.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

# END POINT
@api_view(['POST'])
def create_user(request):
    # Ubah data jadi JSON
    serializer = UserSerializer(data=request.data)
    if(serializer.is_valid()):
        # Simpan data di DB
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED )
    else:
        return Response(status=status.HTTP_404_NOT_FOUND)

# END POINT
@api_view(['GET', 'PUT', 'DELETE'])
def user_details(request, pk):
    try:
        # Ambil data berdasarkan PK
        User = user.objects.get(pk=pk)
    # Kalo PK tidak ada di DB
    except user.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
        # konversi data ke JSON
        serializer = UserSerializer(User)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        # Cek jika data sesua format dan ubah ke json
        serializer = UserSerializer(User, data=request.data)
        if serializer.is_valid():
            # Simpan data di DB
            serializer.save()
            return Response(serializer.data)
        # Jika data tidak valid / error
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        # Hapus data dari DB
        User.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
